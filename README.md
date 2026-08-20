# Advanced RAG Architectures with LangGraph: Corrective RAG & Self-RAG

Two RAG (Retrieval-Augmented Generation) systems built from scratch, step by step, using **LangGraph** to implement self-correcting retrieval pipelines instead of plain single-pass RAG. Each system is broken into incremental notebooks so the reasoning behind every added node is visible, not just the final graph.

---

## 📁 Repository Structure

```
.
├── corrective-rag/
│   ├── 1_basic_rag.ipynb
│   ├── 2_retrieval_refinement.ipynb
│   ├── 3_retrieval_evaluator.ipynb
│   ├── 4_web_search_refinement.ipynb
│   ├── 5_query_rewrite.ipynb
│   ├── 6_ambiguous.ipynb
│   └── documents/            # sample PDFs used for retrieval
│
├── self-rag/
│   ├── self_rag_step1.ipynb  → self_rag_step7.ipynb
│   ├── self_rag_web.ipynb
│   └── documents/            # sample PDFs used for retrieval
│
└── README.md
```

---

## 1️⃣ Corrective RAG (CRAG)

A retrieval pipeline that **grades its own retrieved documents** and dynamically decides whether to trust them, rewrite the query, or fall back to a web search — instead of blindly stuffing whatever the retriever returns into the prompt.

**Flow logic:**
1. `decide_retrieval` — routes straight to a direct LLM answer if no retrieval is needed, otherwise proceeds to `retrieve`.
2. `retrieve` — pulls candidate chunks from a FAISS vector store built over the uploaded PDFs.
3. `is_relevant` — an LLM-based grader scores whether the retrieved chunks actually answer the query.
4. If relevant → `generate_from_context` → `is_sup` checks the answer is **grounded/supported** by the retrieved context; if not, it loops into `revise_answer` before re-checking.
5. If **not** relevant → `rewrite_question` reformulates the query and retries retrieval (bounded loop).
6. `is_use` performs a final usefulness check on the generated answer before ending, falling back to `no_answer_found` if nothing usable was produced.

*(See `docs/corrective-rag-flow.png` for the full graph diagram.)*

### Notebooks (built incrementally)
| Notebook | What it adds |
|---|---|
| `1_basic_rag.ipynb` | Baseline RAG: PDF ingestion → chunking → FAISS retrieval → generation |
| `2_retrieval_refinement.ipynb` | Structured state (TypedDict) + Pydantic-based routing logic |
| `3_retrieval_evaluator.ipynb` | LLM-as-judge grader node to score retrieved document relevance |
| `4_web_search_refinement.ipynb` | Tavily web search fallback when local retrieval is insufficient |
| `5_query_rewrite.ipynb` | Query rewriting node to improve retrieval on low-relevance results |
| `6_ambiguous.ipynb` | Handling ambiguous/unanswerable queries gracefully |

---

## 2️⃣ Self-RAG

A retrieval pipeline modeled on the **Self-RAG paper**, where the LLM critiques its own generations at multiple stages — evaluating document relevance per-chunk, checking support/groundedness, and deciding when to fall back to web search — rather than relying on a single fixed grading step.

**Flow logic:**
1. `retrieve` — fetches candidate chunks from FAISS.
2. `eval_each_doc` — grades **each retrieved document individually** (not the batch as a whole).
3. If retrieval is weak → `rewrite_query` → `web_search` → `refine`, blending fresh web results back into context.
4. If retrieval is sufficient → straight to `refine`.
5. `refine` consolidates the best-supported context, then routes to either `generate` (produce grounded answer) or `ambiguous` (flag as unanswerable) before ending.

*(See `docs/self-rag-flow.png` for the full graph diagram.)*

### Notebooks (built incrementally)
| Notebook | What it adds |
|---|---|
| `self_rag_step1.ipynb` | Baseline retrieval + generation graph |
| `self_rag_step2.ipynb` – `step3.ipynb` | Per-document relevance grading |
| `self_rag_step4.ipynb` – `step5.ipynb` | Groundedness/support checking against retrieved context |
| `self_rag_step6.ipynb` – `step7.ipynb` | Answer refinement and ambiguity handling |
| `self_rag_web.ipynb` | Tavily web search integration as a retrieval fallback |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph (`StateGraph`, conditional edges) |
| LLM & Embeddings | LangChain (`ChatOpenAI`, `OpenAIEmbeddings`) |
| Vector Store | FAISS |
| Document Loading | `PyPDFLoader` + `RecursiveCharacterTextSplitter` |
| Web Search Fallback | Tavily Search API |
| Structured Routing | Pydantic models + `TypedDict` state schemas |
| Config | `python-dotenv` |

---

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

> ⚠️ Never commit real API keys. Keep `.env` in `.gitignore`.

Run any notebook with Jupyter:
```bash
jupyter notebook
```
Each notebook is self-contained and numbered — start at step 1 in either folder to see the pipeline built up incrementally.

---

## 💡 Why Corrective RAG & Self-RAG (over plain RAG)

Plain RAG blindly trusts whatever the retriever returns. Both architectures here add a **feedback loop**:
- **CRAG** grades retrieval quality *before* generating, and corrects course (rewrite query / go to web) when retrieval is weak.
- **Self-RAG** grades *per-document* relevance and checks the *generated answer's* groundedness against context, catching hallucinations that pure retrieval-grading would miss.

This project was built to understand these correction loops from first principles — each notebook adds exactly one new capability on top of the last, rather than jumping straight to the final graph.

---

## 🔮 Future Improvements

- [ ] Merge both pipelines into a single configurable LangGraph app (select CRAG vs Self-RAG at runtime)
- [ ] Add a Streamlit/Gradio front-end for interactive querying
- [ ] Add automated eval harness (e.g., RAGAS) to compare CRAG vs Self-RAG answer quality
- [ ] Cache FAISS indices to disk instead of rebuilding per run
- [ ] Add unit tests for grading/routing nodes

---

## 📄 License

MIT License
