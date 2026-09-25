import { test, expect } from "vitest";
import { createHash } from "node:crypto";
import init, { LocalIndex } from "../pkg/summa_wasm";
import { InMemoryFS } from "./storage.ts";
import simdFixture from "./fixtures/simd4x.json";
import impactFixture from "./fixtures/impacts.json";
import compactFixture from "./fixtures/compact-simd4x.json";
import groupFixture from "./fixtures/impact-groups.json";

// Native ARM SIMD-produced bytes must decode identically through the library's
// WASM scalar backend, including full blocks, rare-term tails and positions.
// The "rounded" control in every fixture is the rounded codec written with
// --posting-ratio-bounds (scripts/search_benchmark/wasm_codec_fixture.py); the
// variant under test differs only in codec (simd4x) or bound layout (impacts).
test.each([["SIMD", simdFixture], ["impact", impactFixture], ["group impacts", groupFixture], ["compact SIMD and byte norms", compactFixture]])("native %s index bytes preserve exact browser scores and phrase hits", async (_, fixture) => {
  await init();
  const controls: unknown[] = [];
  for (const [codec, data] of Object.entries(fixture.indexes)) {
    const storage = new InMemoryFS();
    for (const [name, encoded] of Object.entries(data.files)) {
      const bytes = Uint8Array.from(Buffer.from(encoded, "base64"));
      expect(createHash("sha256").update(bytes).digest("hex")).toBe(data.sha256[name]);
      await storage.write(name, bytes.buffer);
    }
    const index = await LocalIndex.withStorage(storage, "index fixture { field text: text }");
    expect(index.numDocs()).toBe(fixture.documents);
    // Both common terms can become score-required after the top-k heap fills.
    // Requesting every document supplies an unpruned browser oracle too.
    for (const [i, query] of [...fixture.queries, "alpha OR beta", "alpha AND beta"].entries()) {
      const response = await index.search(query, fixture.documents);
      expect(response.hits.length).toBe(data.counts[i] ?? fixture.documents);
      const normalized = await Promise.all(response.hits.map(async (hit: any) => {
        const doc = await index.getDocument(hit.address.segment_id, hit.address.doc_id);
        return [doc.id, hit.score];
      }));
      normalized.sort((a, b) => String(a[0]).localeCompare(String(b[0])));
      const top = await index.search(query, 10);
      expect(top.hits).toEqual(response.hits.slice(0, 10));
      if (codec === "rounded") controls.push(normalized); // rounded + ratio bounds control
      else expect(normalized).toEqual(controls[i]);
    }
    index.free();
  }
});
