import { test, expect } from "vitest";
import init, { LocalIndex } from "../pkg/summa_wasm";

test("phrase selectivity preserves matches and ranking with frequent terms at every offset", async () => {
  await init();
  const index = await LocalIndex.create("index selective_phrases { field id: u64 [stored] field text: text [indexed<token_position>] }");
  const phrases = ["common middle rare", "middle common rare", "common rare unique", "rare common middle", "common common rare"];
  const documents = Array.from({ length: 700 }, (_, id) => {
    const tokens = Array(30 + id % 7).fill("common");
    if (id % 2 === 0) tokens.push("middle");
    if (id % 7 === 0) tokens.push("rare");
    if (id % 31 === 0) tokens.push("unique");
    for (let i = 0; i < phrases.length; i++) {
      if (id % (17 + i * 2) === 0) tokens.push("gap", ...phrases[i].split(" "), "gap");
    }
    return { id, text: tokens.join(" ") };
  });
  try {
    await index.addDocuments(documents);
    await index.commit();
    for (const phrase of phrases) {
      const terms = phrase.split(" ");
      const expected = documents.filter(doc => {
        const tokens = doc.text.split(" ");
        return tokens.some((_, start) => terms.every((term, offset) => tokens[start + offset] === term));
      }).map(doc => doc.id);
      const all = await index.search(`"${phrase}"`, documents.length);
      const ids = await Promise.all(all.hits.map(async (hit: any) =>
        (await index.getDocument(hit.address.segment_id, hit.address.doc_id)).id,
      ));
      expect(ids.sort((a, b) => a - b)).toEqual(expected);
      for (const limit of [1, 10, 100]) {
        expect((await index.search(`"${phrase}"`, limit)).hits).toEqual(all.hits.slice(0, limit));
      }
    }
  } finally {
    index.free();
  }
});
