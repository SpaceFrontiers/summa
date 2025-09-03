import { test, expect } from "vitest";

import init, { LocalIndex } from "../pkg/hermes_wasm";

test("Search in index", async () => {
	await init();

	// Define schema using SDL
	const index = await LocalIndex.create(`
		index articles {
			field title: text<en_stem> [indexed, stored]
			field body:  text<en_stem> [indexed, stored]
			field views: u64 [indexed, stored]
		}
	`);

	// Add documents
	await index.addDocuments([
		{
			title: "Rust Programming",
			body: "Rust is a systems language.",
			views: 1500,
		},
		{
			title: "Search Engines",
			body: "BM25 is a ranking function.",
			views: 800,
		},
	]);

	// Commit (builds the segment)
	await index.commit();
	expect(index.numDocs()).toBe(2);
	expect(index.fieldNames()).toEqual(["title", "body", "views"]);

	// Search
	const results = await index.search("rust", 10);
	// { hits: [{ address: { segment_id, doc_id }, score }], total_hits: 1 }

	// Get document
	const doc = await index.getDocument(
		results.hits[0].address.segment_id,
		results.hits[0].address.doc_id,
	);

	expect(doc).toEqual({
		title: "Rust Programming",
		body: "Rust is a systems language.",
		views: 1500,
	});

	const titleOnly = await index.getDocumentWithFields(
		results.hits[0].address.segment_id,
		results.hits[0].address.doc_id,
		["title"],
	);
	expect(titleOnly).toEqual({ title: "Rust Programming" });
});

test("BMP search returns identical scores with optional forward storage", async () => {
	await init();
	const index = await LocalIndex.create(`
		index forward_test {
			field sparse: sparse_vector [indexed<format: bmp, dims: 32, max_weight: 5.0, bmp_block_size: 8>]
			field inverted: sparse_vector [indexed<format: bmp, dims: 32, max_weight: 5.0, bmp_block_size: 8, bmp_forward_index: false>]
		}
	`);
	await index.addDocuments(Array.from({ length: 64 }, (_, doc) => ({
		sparse: { indices: Array.from({ length: 12 }, (_, dim) => dim), values: Array(12).fill(doc % 8 === 0 ? 1 : 0.02) },
		inverted: { indices: Array.from({ length: 12 }, (_, dim) => dim), values: Array(12).fill(doc % 8 === 0 ? 1 : 0.02) },
	})));
	await index.commit();
	const request = (field: string) => ({ query: { sparseVector: {
		field, indices: Array.from({ length: 12 }, (_, i) => i),
		values: Array.from({ length: 12 }, (_, i) => i < 3 ? 2 : 0.5),
	} }, limit: 1 });
	const baseline = await index.searchStructured(request("sparse"));
	expect(baseline.hits).toHaveLength(1);
	expect((await index.searchStructured(request("inverted"))).hits).toEqual(baseline.hits);
});

test.each([0, 1])("Sparse query language inherits schema LSP gamma %i", async (gamma) => {
	await init();
	const index = await LocalIndex.create(`
		index sparse_policy {
			field emb: sparse_vector [indexed<format: bmp, dims: 16, max_weight: 5.0, bmp_block_size: 1, query<lsp_gamma: ${gamma}>>]
		}
	`);
	// Two superblocks: eight weaker documents followed by one winner.
	await index.addDocuments(Array.from({ length: 9 }, (_, doc) => ({
		emb: { indices: [0], values: [doc === 8 ? 5.0 : 0.1] },
	})));
	await index.commit();
	const results = await index.search("emb:sparse({0: 1.0})", 9);
	expect(results.hits).toHaveLength(gamma === 0 ? 9 : 1);
	expect(results.hits[0].address.doc_id).toBe(8);
});


test("Tracing preserves branch candidates before pagination and RRF attribution preserves ranking", async () => {
    await init();
    const index = await LocalIndex.create(`index traces {
        field title: text [indexed, stored]
        field body: text [indexed, stored]
    }`);
    await index.addDocuments([
        { title: "rust", body: "rust" },
        { title: "rust rust", body: "other rust" },
    ]);
    await index.commit();
    const query = { fusion: { queries: [
        { name: "title", query: { term: { field: "title", value: "rust" } }, weight: 0.7 },
        { name: "body", query: { match: { field: "body", text: "rust" } }, weight: 2 },
    ], rrfK: 42, fetchLimit: 2 } };
    const plain = await index.searchStructured({ query, limit: 1 });
    expect(plain.trace).toBeUndefined();
    expect(plain.hits[0].rrf_score).toBeUndefined();
    const traced = await index.searchStructured({ query, limit: 1, includeRrfScores: true, tracing: true });
    const { rrf_score, rrf_contributions, ...hit } = traced.hits[0];
    expect(hit).toEqual(plain.hits[0]);
    expect(rrf_score).toBe(traced.hits[0].score);
    expect(rrf_contributions.map((vote: any) => vote.query_name)).toEqual(["title", "body"]);
    expect(traced.trace.shards[0].queries.map((q: any) => q.candidates.length)).toEqual([2, 2]);
    expect(traced.trace.shards[0].selected).toHaveLength(1);
    expect(traced.trace.shards[0].queries[0].query.term.field).toBe("title");
    const page = await index.searchStructured({ query, limit: 1, offset: 1, includeRrfScores: true, tracing: true });
    expect(page.hits[0].address).not.toEqual(traced.hits[0].address);
    expect(page.trace.shards[0].queries).toEqual(traced.trace.shards[0].queries);
    expect(page.hits[0].rrf_contributions.some((vote: any) => vote.rank === 2)).toBe(true);
    const rootQuery = { boolean: { must: [{ term: { field: "title", value: "rust" } }] } };
    const root = await index.searchStructured({ query: rootQuery, limit: 1, offset: 1, tracing: true });
    expect(root.trace.shards[0].queries[0].candidates).toHaveLength(2);
    expect(root.trace.shards[0].queries[0].query.boolean.must[0].term.field).toBe("title");
    expect(root.hits).toEqual((await index.searchStructured({ query: rootQuery, limit: 1, offset: 1 })).hits);
    await expect(index.searchStructured({ query: rootQuery, includeRrfScores: true })).rejects.toContain("requires fusion");
});
