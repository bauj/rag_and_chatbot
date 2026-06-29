# Next Steps — Improving Retrieval & Extraction

This document captures research into improving the chatbot's two weak links:

1. **Extraction quality** — how cleanly HTML is turned into text chunks.
2. **Retrieval quality** — how well the right chunks are found for a question.

It explains the jargon, ranks the options by effort vs. impact, and gives concrete
steps grounded in the current codebase.

---

## TL;DR — Recommended order

| # | Change | Where | Effort | Payoff |
|---|--------|-------|--------|--------|
| 1 | **markdownify** instead of `.get_text()` | `extraction/` | ~1 day | Code blocks survive chunking → better embeddings & answers |
| 2 | **BM25 hybrid retrieval** | `chatbot/core/rag_chatbot.py` | ~1 day | Exact class/method-name matching, which embeddings are bad at |
| 3 | **HyDE** | `chatbot/core/rag_chatbot.py` | ~1 day | Better recall on vague/short questions, no re-indexing |
| 4 | **ColBERTv2** (late interaction) | new retriever | 2–3 days | Replaces MiniLM **and** the BGE reranker with one stronger stage |
| 5 | **MinerU-HTML** model extraction | `extraction/` | ~2 days | Higher chunk quality if you have an LLM server already |
| — | **GraphRAG** | — | 1+ week | **Skip** — see reasoning below |

The first two are the "low-hanging fruit": no re-architecture, just incremental
additions. Start there, measure, then decide whether 3–5 are worth it.

---

## Part 1 — Jargon, explained

### BM25 / "sparse" retrieval

**BM25** is a classic keyword-ranking formula (the thing search engines used before
neural nets). It scores a document by *how often the query's words appear in it*,
adjusted so that (a) rare words count more than common ones and (b) very long
documents don't win just by being long. It is called **sparse** retrieval because a
document is represented as a giant mostly-zero vector over the whole vocabulary —
one slot per word, non-zero only for words actually present.

- **Strength:** exact token matching. `Model_FeatureBuilder::addFeature` matches the
  page that literally contains `Model_FeatureBuilder::addFeature`.
- **Weakness:** no notion of meaning. "delete a shape" will not match a page that
  only says "remove a geometry" — different words, same intent.

### Dense / vector retrieval (what you have now)

A **dense** embedding model (your `all-MiniLM-L6-v2`) maps each chunk to a single
fixed-length vector (e.g. 384 numbers) that captures *meaning*. Similar meanings →
nearby vectors. This is the opposite trade-off from BM25:

- **Strength:** semantic matching ("delete a shape" ≈ "remove a geometry").
- **Weakness:** exact symbol names get blurred. `addFeature`, `addNode`, and
  `addFace` all look "similar" to the model and can be confused — bad for API docs
  where the exact identifier is the whole point.

### Hybrid retrieval (BM25 + dense)

**Run both retrievers and merge their results.** The standard merge is **RRF
(Reciprocal Rank Fusion)**: each document gets a score of `1 / (k + rank)` from each
retriever (rank = its position in that retriever's list), and the scores are summed.
A document ranked highly by *either* method floats to the top. This is the de-facto
2025 default for technical-doc retrieval precisely because it covers both the
exact-name case (BM25) and the paraphrase case (dense).

In LangChain this is `EnsembleRetriever` wrapping a `BM25Retriever` and your existing
Chroma retriever — roughly 20 lines.

### HyDE (Hypothetical Document Embeddings)

The problem: a short question ("how to mesh?") embeds poorly because it is nothing
like the long, technical *answer* text it should match. **HyDE** fixes this by asking
the LLM to first *write a fake answer* to the question, then embedding **that** and
searching with it. The hallucinated answer is wrong on details but lands in the right
neighbourhood of the vector space, so retrieval improves. Cost: one extra LLM call
per query, no re-indexing.

### Cross-encoder reranking (what you have now)

Your `BAAI/bge-reranker-v2-m3` is a **cross-encoder**: it takes `(query, chunk)`
*together* and outputs a relevance score. Far more accurate than comparing two
separate vectors, but *slow* — it must run the model once per candidate chunk. That's
why it only runs on the ~40–60 already-retrieved chunks, not the whole DB. See
`_rerank_and_expand()` in `rag_chatbot.py`.

### ColBERTv2 / late interaction

A middle ground between dense retrieval and cross-encoding. Instead of one vector per
chunk, ColBERT stores **one vector per token**. At query time, each *query token*
finds its best-matching *document token* (an operation called **MaxSim**), and those
best-match scores are summed. This keeps much of a cross-encoder's accuracy at close
to dense-retrieval speed, and it's especially good at exact technical terms (because
individual tokens are matched directly).

- It can **replace both** your MiniLM retriever and your BGE reranker with a single
  stage.
- Python: `RAGatouille` was the popular wrapper; maintenance has shifted to
  **`pylate`** (2025) — check that first.
- Friction: it uses its own index format (PLAID), **not** ChromaDB. So your metadata
  filters (`module`, `doc_category`) need separate handling — either pre-filter
  candidates in Chroma then score with ColBERT, or store metadata alongside the
  ColBERT index and filter after retrieval.

### RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

Addresses questions whose answer is **spread across many pages**. At index time it
clusters related chunks, has the LLM **summarize each cluster**, then clusters and
summarizes *those summaries* — building a tree from fine-grained leaves up to
high-level overviews. At query time you can retrieve from any level, so a broad
question ("explain the overall meshing workflow") hits a high-level summary node
instead of 30 disconnected fragments. Moderate indexing cost (LLM calls to build the
tree). Useful mostly for the **Sphinx user-guide** side where cross-page synthesis
matters; less so for API reference.

### GraphRAG (and why to skip it here)

GraphRAG uses an LLM to extract **entities and relationships** from every chunk,
assembles a knowledge graph, detects "communities" of related entities, and writes
summaries of each. Great for *unstructured prose* where relationships are implicit
(reports, wikis).

**Why it's a poor fit for this project:** Doxygen HTML *already encodes* the entity
graph — classes, methods, namespaces, inheritance, cross-references are all explicit
in the markup. GraphRAG would spend many LLM calls re-deriving structure you already
threw away during extraction — a lossy, expensive round-trip. The graph also has to
be rebuilt whenever docs change. Revisit only if you add substantial narrative design
docs. (Watch **LightRAG**/**LazyGraphRAG** as lighter variants if you ever do.)

### Other terms you'll see

- **SPLADE** — a "learned sparse" model: like BM25 but a neural net decides word
  weights and can add related terms. Better than BM25, heavier to run. Only worth it
  if BM25's pure exact-match still misses too much.
- **Contextual Retrieval** (Anthropic) — at index time, prepend an LLM-written
  one-line "what this chunk is about" to each chunk before embedding. Improves
  retrieval; costs tokens at indexing, not at query time.
- **Bi-encoder** — the general name for the single-vector-per-text approach
  (your MiniLM). Contrast with cross-encoder (above).

---

## Part 2 — Step-by-step plans

### Step 1 — Preserve code blocks with markdownify  *(start here)*

**Problem.** Extraction calls `.get_text()` in several places
(`extraction/process_docs.py:257`, `:292`; `extraction/html_parser.py:53`). This
flattens `<pre>`/`<code>` blocks into running prose — indentation, line breaks, and
the visual signal "this is code" are all lost, which hurts both embedding quality and
the readability of retrieved context.

**Fix.** On the already-selected content subtree, convert to Markdown instead of
flattening. `markdownify` turns `<pre>` into fenced ```code blocks``` and preserves
structure.

```bash
pip install markdownify
```

Concrete edits:
1. In `html_parser.py`, add a helper, e.g. `element_to_markdown(element)` that calls
   `markdownify(str(element), code_language="")` (after the existing
   `script`/`style`/`noscript` decompose).
2. Swap the `.get_text(separator='\n', strip=True)` calls in
   `extract_sections_with_soup` / `_split_into_sections` for the Markdown helper so
   section text keeps its code fences.
3. Mirror the change in `process_docs.py`'s `extract_content_doxygen` /
   `extract_content_sphinx`.
4. Re-run extraction and spot-check a class page in
   `{project}_docs.jsonl` — code should now appear as fenced blocks.

**Watch out:** chunk sizes grow slightly (Markdown adds characters). Re-check the
token-chunking path (`chunk_text_by_tokens`) still respects the 512-token limit.

### Step 2 — BM25 hybrid retrieval

**Goal.** Catch exact class/method names that `all-MiniLM-L6-v2` blurs together.

```bash
pip install rank_bm25
```

Plan (in `rag_chatbot.py`):
1. Build a `BM25Retriever` from the same chunks. The simplest source is the
   `{project}_docs.jsonl` already written by extraction — load it once at startup and
   `BM25Retriever.from_texts(texts, metadatas=...)`.
2. In `_create_chain`, wrap both retrievers:
   ```python
   from langchain.retrievers import EnsembleRetriever
   ensemble = EnsembleRetriever(
       retrievers=[chroma_retriever, bm25_retriever],
       weights=[0.5, 0.5],   # tune
   )
   ```
3. Feed the ensemble's output into the existing `_rerank_and_expand` step — hybrid
   *and* reranking compose cleanly.
4. **Filtering caveat:** `BM25Retriever` doesn't apply Chroma's metadata `filter`.
   When a `module`/`doc_type` filter is active, either filter the BM25 candidate list
   in Python afterward, or keep BM25 unfiltered and rely on rerank to sort it out.
   Decide based on how often filters are used.

**Measure:** before/after on a handful of known symbol-name questions
("what does `addFeature` do?"). This is the cheapest way to confirm the gain.

### Step 3 — HyDE

**Goal.** Better recall on short/vague questions, no re-indexing.

Plan (in `rag_chatbot.py`):
1. Before retrieval, add an LLM call: *"Write a short documentation passage that would
   answer: {question}"*.
2. Embed and retrieve with that generated passage instead of (or in addition to) the
   raw question.
3. Keep the **original** question for the final answer-generation prompt — HyDE only
   changes what you *search* with, not what you ask the LLM to answer.
4. Make it toggleable (config flag) — it adds latency and isn't always a win for
   already-specific questions.

### Step 4 — ColBERTv2 (only if 1–3 aren't enough)

**Goal.** Replace MiniLM retrieval **and** the BGE reranker with one stronger
late-interaction stage.

Plan:
1. Evaluate `pylate` first (active), `RAGatouille` second.
2. Build a ColBERT index from the same chunk texts. Expect a separate index
   directory (PLAID format) alongside — or instead of — ChromaDB.
3. Decide the metadata-filtering strategy up front (the main integration cost):
   - **Option A:** keep ChromaDB for a filtered first pass, then ColBERT to score.
   - **Option B:** store `module`/`doc_category` next to the ColBERT index and filter
     post-retrieval.
4. If it wins, you can drop the `reranker` block from config entirely — ColBERT
   subsumes it.

**Note:** this is the biggest architectural change here. Only take it on after 1–3
have been measured and found insufficient.

### Step 5 — MinerU-HTML extraction (optional, model-based)

If markdownify isn't clean enough and you already run a local LLM for the chatbot:
`pip install mineru`. It extracts main content via a small model and supports
OpenAI-compatible backends, so you can point it at your existing `llm.base_url`.
Higher quality than rule-based extraction, but slower and a heavier dependency —
treat as a fallback if Step 1 leaves too much noise. (**ReaderLM v2** from Jina is the
same tier but needs a GPU; only practical for small corpora.)

---

## Part 3 — How to measure (do this before/after each change)

The TODO already mentions LangSmith evaluation — that's the right home for this.
Minimum viable evaluation without it:

1. Hand-write 15–25 representative questions across your modules: some exact-symbol
   ("what params does `X::y` take?"), some conceptual ("how does meshing work?"),
   some cross-page.
2. For each change, record whether the correct source page appears in the retrieved
   set (retrieval recall) and eyeball answer quality.
3. Only keep a change if it helps without regressing the others — hybrid/HyDE can
   occasionally hurt already-precise queries.

Without numbers, every option above is just a plausible guess. The evaluation harness
is the highest-leverage thing to build first if you intend to iterate seriously.
