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

test("committed deletion masks hide indexed-only rows after WASM reopen", async () => {
	await init();
	const storage = new InMemoryFS();
	const schema = "index documents { field id: text<raw> [stored] field body: text<simple> [indexed] }";
	const index = await LocalIndex.withStorage(storage, schema);
	await index.addDocuments([
		{ id: "deleted", body: "needle" },
		{ id: "live-a", body: "needle" },
		{ id: "live-b", body: "needle" },
	]);
	await index.commit();
	const original = await index.search("needle", 10);
	const metadata = JSON.parse(new TextDecoder().decode(await storage.get("metadata.json")));
	const segment = Object.keys(metadata.segment_metas)[0];
	const maskId = "11111111111111111111111111111111";
	const mask = new Uint8Array(32);
	mask.set([72, 68, 69, 76, 1, 0, 0, 0]);
	const view = new DataView(mask.buffer);
	view.setUint32(8, 3, true);
	view.setUint32(12, 1, true);
	view.setBigUint64(16, 6n, true); // physical rows one and two stay live
	let hash = 0xcbf29ce484222325n;
	for (const byte of mask.slice(0, 24)) hash = BigInt.asUintN(64, (hash ^ BigInt(byte)) * 0x100000001b3n);
	view.setBigUint64(24, hash, true);
	await storage.write(`seg_${maskId}.del`, mask.buffer);
	metadata.segment_metas[segment].deletions = { id: maskId, num_deleted: 1 };
	metadata.publication_generation += 1;
	await storage.write("metadata.json", new TextEncoder().encode(JSON.stringify(metadata)).buffer);
	const reopened = await LocalIndex.withStorage(storage, schema);
	expect(reopened.numDocs()).toBe(2);
	expect((await reopened.search("needle", 2)).hits.map((hit: any) => hit.address.doc_id).sort()).toEqual([1, 2]);
	expect(await reopened.getDocument(original.hits[0].address.segment_id, 0)).toBeNull();
	expect((await index.search("needle", 10)).hits).toHaveLength(3);
	mask[16] ^= 3; // preserve population but invalidate the checksum
	await storage.write(`seg_${maskId}.del`, mask.buffer);
	await expect(LocalIndex.withStorage(storage, schema)).rejects.toThrow(/deletion/);
});

const mutationSchema = `index documents {
  field id: text<raw> [primary, indexed, stored]
  field body: text<simple> [indexed<chunked, token_position>]
}`;

test("primary-key mutations hide every indexed-only chunk and survive reopen", async () => {
  await init();
  const storage = new InMemoryFS();
  const index = await LocalIndex.withStorage(storage, mutationSchema);
  await index.addDocuments([
    { id: "a", body: ["oldhead needle", "oldtail needle"] },
    { id: "b", body: ["keep needle", "keep tail"] },
  ]);
  await index.commit();
  const old = (await index.search("oldtail", 10)).hits[0].address;
  await expect(index.addDocument({ id: "a", body: "duplicate" })).rejects.toThrow(/uplicate/);
  await index.upsertDocument({ id: "a", body: ["newhead", "newtail"] });
  expect((await index.search("oldtail", 10)).hits).toHaveLength(1);
  expect(() => index.deleteDocument("a")).toThrow(/pending insertion/);
  const deletion = index.deleteDocuments(["b", "absent", ""]);
  expect(deletion.acceptedCount).toBe(2);
  expect(deletion.errors.map((error: any) => error.index)).toEqual([2]);
  await index.commit();
  expect(index.numDocs()).toBe(1);
  expect((await index.search("needle", 10)).hits).toHaveLength(0);
  expect((await index.search("newtail", 10)).hits).toHaveLength(1);
  expect(await index.getDocument(old.segment_id, old.doc_id)).toBeNull();
  const reopened = await LocalIndex.withStorage(storage, mutationSchema);
  await expect(reopened.addDocument({ id: "a", body: "duplicate" })).rejects.toThrow(/uplicate/);
  await reopened.addDocument({ id: "b", body: "reused" });
  await reopened.commit();
  expect(reopened.numDocs()).toBe(2);
  expect((await reopened.search("oldhead", 10)).hits).toHaveLength(0);
});

test("failed and aborted replacements preserve old rows and reservations", async () => {
  await init();
  const index = await LocalIndex.create(mutationSchema);
  await index.addDocument({ id: "a", body: "original" });
  await index.commit();
  const response = await index.upsertDocuments([
    { body: "missing key" },
    { id: "a", body: "replacement" },
    { id: "a", body: "second replacement" },
    { id: ["x", "y"], body: "ambiguous" },
  ]);
  expect(response.acceptedCount).toBe(1);
  expect(response.errors.map((error: any) => error.index)).toEqual([0, 2, 3]);
  await index.abort();
  expect(await index.commit()).toBe(false);
  expect((await index.search("original", 10)).hits).toHaveLength(1);
  await index.upsertDocument({ id: "a", body: "committed" });
  await index.commit();
  expect((await index.search("committed", 10)).hits).toHaveLength(1);
  expect((await index.search("original", 10)).hits).toHaveLength(0);
  expect(() => index.deleteDocuments(new Array(100001).fill("a"))).toThrow(/exceeds/);
  await expect(index.upsertDocuments(new Array(1001).fill({ id: "a" }))).rejects.toThrow(/exceeds/);
});

test.each([false, true])("storage commit retry after metadata write=%s does not replay deletions", async (afterWrite) => {
  await init();
  const storage = new InMemoryFS();
  const write = storage.write;
  let fail = false;
  storage.write = async (name, data) => {
    if (afterWrite || !fail || name !== "metadata.json") await write(name, data);
    if (fail && name === "metadata.json") { fail = false; throw new Error("injected persistence failure"); }
  };
  const index = await LocalIndex.withStorage(storage, mutationSchema);
  await index.addDocument({ id: "a", body: "original" });
  await index.commit();
  await index.upsertDocument({ id: "a", body: "replacement" });
  fail = true;
  await expect(index.commit()).rejects.toThrow(/injected/);
  const previous = await LocalIndex.withStorage(storage, mutationSchema);
  expect((await previous.search(afterWrite ? "replacement" : "original", 10)).hits).toHaveLength(1);
  expect(await index.commit()).toBe(true);
  expect(await index.commit()).toBe(false);
  const reopened = await LocalIndex.withStorage(storage, mutationSchema);
  expect(reopened.numDocs()).toBe(1);
  expect((await reopened.search("replacement", 10)).hits).toHaveLength(1);
  expect((await reopened.search("original", 10)).hits).toHaveLength(0);
  reopened.deleteDocument("a");
  await reopened.commit();
  expect(reopened.numDocs()).toBe(0);
  await reopened.addDocument({ id: "a", body: "reused" });
  await reopened.commit();
  expect(reopened.numDocs()).toBe(1);
});

test("replacement batches preserve JS integer conversion", async () => {
  await init();
  const schema = mutationSchema.replace("field body:", "field n: u64 [stored]\n field body:");
  const index = await LocalIndex.create(schema);
  const response = await index.upsertDocuments([{ id: "a", body: "integer", n: 13n }]);
  expect(response.acceptedCount).toBe(1);
  expect(response.errors).toEqual([]);
  await index.commit();
  const address = (await index.search("integer", 10)).hits[0].address;
  expect((await index.getDocument(address.segment_id, address.doc_id)).n).toBe(13);
});

test("reopen reclaims orphan outputs after an interrupted storage commit", async () => {
  await init();
  const storage = new InMemoryFS();
  const write = storage.write;
  let fail = false;
  storage.write = async (name, data) => {
    if (fail && name === "metadata.json") throw new Error("interrupted publication");
    await write(name, data);
  };
  const index = await LocalIndex.withStorage(storage, mutationSchema);
  await index.addDocument({ id: "a", body: "original" });
  await index.commit();
  const before = (await storage.list()).sort();
  fail = true;
  await index.upsertDocument({ id: "a", body: "unpublished" });
  await expect(index.commit()).rejects.toThrow(/interrupted/);
  expect((await storage.list()).length).toBeGreaterThan(before.length);
  index.free();
  storage.write = write;
  const reopened = await LocalIndex.withStorage(storage, mutationSchema);
  expect((await reopened.search("original", 10)).hits).toHaveLength(1);
  await reopened.commit();
  expect((await storage.list()).sort()).toEqual(before);
  expect((await reopened.search("unpublished", 10)).hits).toHaveLength(0);
});
