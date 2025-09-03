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
