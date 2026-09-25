import { test, expect } from "vitest";
import init, { LocalIndex } from "../pkg/summa_wasm";

test("unqualified browser search uses each field's tokenizer in either field order", async () => {
  await init();
  const fields = [
    "field en: text<en_stem> [indexed, stored]",
    "field zh: text<lex(segmenter: unicode)> [indexed, stored]",
  ];
  for (const order of [fields, [...fields].reverse()]) {
    const index = await LocalIndex.create(`index multilingual { ${order.join(" ")} }`);
    try {
      await index.addDocuments([
        {
          en: "The quick brown fox jumps over the lazy dog",
          zh: "那只敏捷的棕毛狐狸跃过了那只懒狗",
        },
        { en: "unrelated", zh: "无关" },
      ]);
      await index.commit();
      for (const query of ["zh:棕毛狐狸", "棕毛狐狸", "棕毛狐狸,"]) {
        const result = await index.search(query, 10);
        expect(result.hits.map((hit: any) => hit.address.doc_id)).toEqual([0]);
      }
    } finally {
      index.free();
    }
  }
});
