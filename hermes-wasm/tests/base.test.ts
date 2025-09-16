import { test, expect } from "vitest";

import init, { LocalIndex } from "../pkg/hermes_wasm";

test("Small limits apply filters before required body candidates", async () => {
	await init();
	const index = await LocalIndex.create(`index filtered_body {
		field kind: text<raw> [indexed, fast]
		field body: text<simple> [indexed<chunked, token_position>]
	}`);
	await index.addDocuments([
		...Array.from({ length: 10 }, () => ({ kind: "article", body: ["machine"] })),
		{ kind: "book", body: ["machine padding padding padding"] },
	]);
	await index.commit();
	for (const query of [
		"body:machine AND kind:book",
		"body:machine AND NOT kind:article",
		"(body:machine OR body:absent) AND kind:book",
	]) {
		const all = await index.search(query, 20);
		expect(all.hits.map((hit: any) => hit.address.doc_id)).toEqual([10]);
		expect((await index.search(query, 1)).hits).toEqual(all.hits);
	}
});

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

test("Hybrid branches retain fast-only type matches and honor exclusions", async () => {
	await init();
	const index = await LocalIndex.create(`index fast_type {
		field type: text<raw_ci> [fast]
		field title: text<simple> [indexed]
		field body: text<simple> [indexed<chunked, token_position>]
	}`);
	await index.addDocuments([
		...Array.from({ length: 3 }, () => ({ type: "journal-article", title: "candidate", body: ["candidate"] })),
		{ type: "book", title: "candidate", body: ["candidate"] },
	]);
	await index.commit();
	const kind = { term: { field: "type", value: "journal-article" } };
	expect((await index.searchStructured({ query: kind, limit: 3 })).hits.map((hit: any) => hit.address.doc_id)).toEqual([0, 1, 2]);
	for (const exclude of [false, true]) {
		const query = { fusion: { fetchLimit: 4, queries: ["title", "body"].map(field => ({
			name: field,
			query: { boolean: {
				must: [{ term: { field, value: "candidate" } }, ...(exclude ? [] : [kind])],
				mustNot: exclude ? [kind] : [],
			} },
		})) } };
		const response = await index.searchStructured({ query, limit: 3, tracing: true });
		expect(response.hits.map((hit: any) => hit.address.doc_id)).toEqual(exclude ? [3] : [0, 1, 2]);
		expect(response.trace.shards[0].queries.map((branch: any) => branch.candidates.length)).toEqual(exclude ? [1, 1] : [3, 3]);
	}
});

test("Seismic preserves top scores for constant-valued Float32 and UInt8 vectors", async () => {
	await init();
	const index = await LocalIndex.create(`
		index forward_test {
			field sparse: sparse_vector [indexed<dims: 32, quantization: float32>]
			field inverted: sparse_vector [indexed<dims: 32, quantization: uint8>]
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

test.each([false, true])("Sparse query language inherits schema exhaustive policy %s", async (exhaustive) => {
	await init();
	const index = await LocalIndex.create(`
		index sparse_policy {
			field emb: sparse_vector [indexed<format: seismic, dims: 16, seismic_postings: 1, seismic_cluster_size: 1, query<exhaustive: ${exhaustive}>>]
		}
	`);
	// Only the strongest row is nominated; exhaustive scans retain all matches.
	await index.addDocuments(Array.from({ length: 9 }, (_, doc) => ({
		emb: { indices: [0], values: [doc === 8 ? 5.0 : 0.1] },
	})));
	await index.commit();
	const results = await index.search("emb:sparse({0: 1.0})", 9);
	expect(results.hits).toHaveLength(exhaustive ? 9 : 1);
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


test("Seismic portable search preserves signed large dimensions and explicit exhaustive overrides", async () => {
    await init();
    const index = await LocalIndex.create(`index wide_sparse {
        field emb: sparse_vector [indexed<format: seismic, dims: 100000, seismic_postings: 1,
            seismic_cluster_size: 1, query<exhaustive: true>>]
    }`);
    await index.addDocuments([
        { emb: { indices: [70000], values: [-4] } },
        { emb: { indices: [70000], values: [-2] } },
        {},
    ]);
    await index.commit();
    const query = { field: "emb", indices: [70000], values: [-1] };
    const exact = await index.searchStructured({ query: { sparseVector: query }, limit: 10 });
    expect(exact.hits.map((hit: any) => hit.score)).toEqual([4, 2]);
    const approximate = await index.searchStructured({
        query: { sparseVector: { ...query, exhaustive: false } }, limit: 10,
    });
    expect(approximate.hits.map((hit: any) => hit.score)).toEqual([4]);
});


test.each(["", "format: maxscore,", "format: seismic,"])("All sparse backends support portable build and query: %s", async (format) => {
    await init();
    const index = await LocalIndex.create(`index sparse_backends {
        field emb: sparse_vector [indexed<${format} dims: 16>]
    }`);
    await index.addDocuments([
        { emb: { indices: [1], values: [1] } },
        { emb: { indices: [1], values: [4] } },
        {},
    ]);
    await index.commit();
    const result = await index.searchStructured({ query: { sparseVector: {
        field: "emb", indices: [1], values: [1], exhaustive: true,
    } }, limit: 10 });
    expect(result.hits.map((hit: any) => hit.address.doc_id)).toEqual([1, 0]);
});

test("Compressed Seismic forward rows preserve portable scoring for groups, tails and U32 fallback", async () => {
    await init();
    const index = await LocalIndex.create(`index compressed_sparse {
        field raw: sparse_vector [indexed<format: seismic, dims: 100000, quantization: float32, seismic_forward_compression: false>]
        field compact: sparse_vector [indexed<format: seismic, dims: 100000, quantization: float32>]
    }`);
    const vectors = [
        { indices: Array.from({ length: 137 }, (_, i) => 65500 + i * 3), values: Array.from({ length: 137 }, (_, i) => i % 11 - 5) },
        { indices: [6, 70000], values: [2, -4] },
        { indices: [6], values: [3] },
    ];
    await index.addDocuments(vectors.map(vector => ({ raw: vector, compact: vector })));
    await index.commit();
    for (const exhaustive of [false, true]) {
        for (const indices of [[65500, 65506], [70000], [65506, 65506], [500]]) {
            const query = (field: string) => ({ query: { sparseVector: {
                field, indices, values: indices.map((_, i) => i % 2 === 0 ? -1 : 1), exhaustive,
            } }, limit: 10 });
            const raw = await index.searchStructured(query("raw"));
            expect((await index.searchStructured(query("compact"))).hits).toEqual(raw.hits);
        }
    }
});

test("BMP packet storage preserves portable retrieval across wide dimensions and packet tails", async () => {
    await init();
    const index = await LocalIndex.create(`index bmp_packets {
        field compact: sparse_vector [indexed<format: bmp, dims: 100000>]
        field inverted: sparse_vector [indexed<format: bmp, dims: 100000, bmp_forward_index: false>]
    }`);
    const vectors = [
        { indices: Array.from({ length: 137 }, (_, i) => 65530 + i * 3), values: Array(137).fill(1) },
        { indices: [6, 70000], values: [2, 3] },
        { indices: [70000], values: [4] },
    ];
    await index.addDocuments([...vectors.map(vector => ({ compact: vector, inverted: vector })), {}]);
    await index.commit();
    for (const indices of [[65530, 65536, 65938], [70000], [70000, 70000], [500]]) {
        const query = (field: string) => ({ query: { sparseVector: {
            field, indices, values: indices.map(() => 1),
        } }, limit: 10 });
        expect((await index.searchStructured(query("compact"))).hits)
            .toEqual((await index.searchStructured(query("inverted"))).hits);
    }
});
