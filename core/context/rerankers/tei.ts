import fetch from "node-fetch";
import { Chunk, Reranker } from "../../index.js";

export class HuggingFaceTEIReranker implements Reranker {
  name = "huggingface-tei";

  static defaultOptions = {
    apiBase: "http://localhost:8080",
    model: "tei",
  };

  constructor(
    private readonly params: {
      apiBase?: string;
      model?: string;
    },
  ) {}

  async rerank(query: string, chunks: Chunk[]): Promise<number[]> {
    let apiBase = this.params.apiBase ?? HuggingFaceTEIReranker.defaultOptions.apiBase;
    if (!apiBase.endsWith("/")) {
      apiBase += "/";
    }

    const resp = await fetch(new URL("rerank", apiBase), {
      method: "POST",
      body: JSON.stringify({
        query: query,
        return_text: false,
        raw_scores: false,
        texts: chunks.map((chunk) => chunk.content),
      }),
    });

    if (!resp.ok) {
      throw new Error(await resp.text());
    }

    const data = (await resp.json()) as any;
    const results = data.results.sort((a: any, b: any) => a.index - b.index);
    return results.map((result: any) => result.score);
  }
}
