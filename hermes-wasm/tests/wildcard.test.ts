import { test, expect } from "vitest";
import init, { LocalIndex } from "../pkg/hermes_wasm";

test("browser wildcard queries match whole Unicode terms and escaped operators", async () => {
  await init();
  const index = await LocalIndex.create("index patterns { field id: u64 [stored] field term: text<raw> [indexed] }");
  const terms = ["there", "theme", "the", "them", "é", "🦀", "a*b", "a?b", "xtherey"];
  try {
    await index.addDocuments(terms.map((term, id) => ({ id, term })));
    await index.commit();
    for (const [pattern, expected] of [
      ["th*e", [0, 1, 2]], ["*ere", [0]], ["?", [4, 5]],
      ["a\\*b", [6]], ["a\\?b", [7]], ["absent*", []],
    ] as const) {
      const results = await index.search(`term:wildcard(${JSON.stringify(pattern)})`, 20);
      const ids = await Promise.all(results.hits.map(async (hit: any) =>
        (await index.getDocument(hit.address.segment_id, hit.address.doc_id)).id,
      ));
      expect(ids.sort((a, b) => a - b)).toEqual(expected);
    }
    expect((await index.search("term:th*e", 20)).hits).toHaveLength(3);
    expect((await index.search("term:*ere", 20)).hits).toHaveLength(1);
    await expect(index.search(`term:wildcard(${JSON.stringify("broken\\")})`, 10))
      .rejects.toMatch(/escape/);
  } finally {
    index.free();
  }
});
