import { test, expect } from "vitest";
import init, { LocalIndex } from "../pkg/summa_wasm";

test("exact browser phrases preserve occurrences across rejected prefixes and repeated terms", async () => {
  await init();
  const index = await LocalIndex.create("index phrases { field id: u64 [stored] field text: text [indexed<token_position>] }");
  const documents = Array.from({ length: 260 }, (_, id) => ({
    id,
    text: Array(1 + id % 5).fill(id % 7 === 0 ? "alpha noise beta gamma alpha alpha" : "alpha beta gamma alpha alpha").join(" "),
  }));
  try {
    await index.addDocuments(documents);
    await index.commit();
    for (const phrase of ["alpha beta gamma", "beta gamma", "alpha alpha", "alpha beta gamma alpha"]) {
      const terms = phrase.split(" ");
      const expected = documents.filter(doc => {
        const tokens = doc.text.split(" ");
        return tokens.some((_, start) => terms.every((term, offset) => tokens[start + offset] === term));
      }).map(doc => doc.id);
      const query = `"${phrase}"`;
      const all = await index.search(query, documents.length);
      const ids = await Promise.all(all.hits.map(async (hit: any) =>
        (await index.getDocument(hit.address.segment_id, hit.address.doc_id)).id,
      ));
      expect(ids.sort((a, b) => a - b)).toEqual(expected);
      for (const limit of [1, 10, 100]) {
        expect((await index.search(query, limit)).hits).toEqual(all.hits.slice(0, limit));
      }
    }
  } finally {
    index.free();
  }
});


test.each(["indexed", "indexed<ordinal>"])("browser phrases reject fields with %s but no token positions", async mode => {
  await init();
  const index = await LocalIndex.create(`index unsupported { field text: text [${mode}] }`);
  try {
    await index.addDocuments([{ text: "alpha gap beta" }, { text: "alpha beta" }]);
    await index.commit();
    await expect(index.search('"alpha beta"', 10)).rejects.toMatch(/token positions/);
    await expect(index.searchStructured({ query: { phrase: { field: "text", text: "alpha beta" } }, limit: 10 })).rejects.toMatch(/token positions/);
    expect((await index.search("alpha AND beta", 10)).hits).toHaveLength(2);
    expect((await index.search('"alpha"', 10)).hits).toHaveLength(2);
  } finally {
    index.free();
  }
});
