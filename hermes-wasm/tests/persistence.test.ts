import { test, expect } from "vitest";

import init, { LocalIndex } from "../pkg/hermes_wasm";
import { InMemoryFS } from "./storage.ts";

const sharedStorage = new InMemoryFS();

test.each([undefined, 256])(
	"L1 phrase cap %s persists across reopen and commit",
	async (cap) => {
		await init();
		const storage = new InMemoryFS();
		const defaultSchema =
			"index documents { field body: text<simple> [indexed<token_position>, stored] }";
		const schema = cap === undefined
			? defaultSchema
			: defaultSchema.replace("{", `{ max_l1_phrase_terms: ${cap}`);
		const index = await LocalIndex.withStorage(storage, schema);
		await index.addDocuments([{ body: "first document" }]);
		await index.commit();
		const metadata = JSON.parse(
			new TextDecoder().decode(await storage.get("metadata.json")),
		);
		expect(metadata.schema.max_l1_phrase_terms).toBe(cap);

		// Reopen uses metadata even when the supplied creation schema omits the cap.
		const reopened = await LocalIndex.withStorage(storage, defaultSchema);
		await reopened.addDocuments([{ body: "second document" }]);
		await reopened.commit();
		const reloaded = JSON.parse(
			new TextDecoder().decode(await storage.get("metadata.json")),
		);
		expect(reloaded.schema.max_l1_phrase_terms).toBe(cap);
		expect((await reopened.search("document", 10)).hits).toHaveLength(2);
	},
);

test("Zero L1 phrase caps fail WASM index creation", async () => {
	await init();
	await expect(LocalIndex.create(
		"index documents { max_l1_phrase_terms: 0 field body: text }",
	)).rejects.toThrow("positive 32-bit integer");
});

test("Fill the index with the data", async () => {
	await init();

	// Define schema using SDL
	const index = await LocalIndex.withStorage(sharedStorage, `
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
});

test("Each instance of Index have its own state", async () => {
	await init();

	// Define schema using SDL
	const index = await LocalIndex.withStorage(new InMemoryFS(), `
		index articles {
			field title: text<en_stem> [indexed, stored]
			field body:  text<en_stem> [indexed, stored]
			field views: u64 [indexed, stored]
		}
	`);

	// No data in this instance
	await expect(index.search("rust", 10)).rejects.toThrow("No committed data");
});

test("Load data in new instance", async () => {
	await init();

	// Define schema using SDL
	const index = await LocalIndex.withStorage(sharedStorage, `
		index articles {
			field title: text<en_stem> [indexed, stored]
			field body:  text<en_stem> [indexed, stored]
			field views: u64 [indexed, stored]
		}
	`);

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
});
