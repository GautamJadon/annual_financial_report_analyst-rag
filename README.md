# Financial Document Analyst (local RAG)

Ask questions about annual reports, financial statements and audit reports.
Everything runs on your laptop (LM Studio): no API keys, no data leaves your machine.

## Setup (5 minutes)

1. Install LM Studio: https://lmstudio.ai
2. In the Discover tab, download:
   - an embedding model: `nomic-embed-text-v1.5`
   - a chat model: `Llama 3.2 3B Instruct` (8 GB RAM) or `Qwen2.5 7B Instruct` (16 GB RAM)
3. Developer tab: load the chat model (set context length to 8192), load the embedding model too, then click **Start Server** (default http://localhost:1234).
4. Install and run:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
5. Upload a text-based annual report PDF, click **Index uploaded PDFs**, then ask questions.

If the embedding model's identifier in LM Studio differs, set it before running:
```
set EMBED_MODEL=your-embedding-model-id      (Windows)
export EMBED_MODEL=your-embedding-model-id   (Mac/Linux)
```
If you indexed with a different embedding model earlier, delete the `chroma_db` folder and re-index.

## Good test questions

- What are the key risk factors mentioned?
- What was total revenue and net profit this year vs last year?
- Is there any qualification or emphasis of matter in the auditor's report?
- List the related party transactions.

## How it works

PDF -> PyMuPDF (spread-aware text + tables as markdown) -> chunks -> embedding model -> ChromaDB
Question -> embed -> top-k chunks (optional per-document filter) -> LM Studio LLM -> answer with [doc p.X] citations

## Ratio calculator (📈 Ratios view)

- Balance sheet and P&L are extracted **by code, not by the LLM** (standalone and consolidated), in about 3 seconds. This also runs automatically when you click *Index uploaded PDFs*.
- Every line item shows its source page and is editable. Eight cross-checks (for example Assets = Equity + Liabilities) flag extraction mistakes.
- 15 ratios (liquidity, leverage, profitability, efficiency) are computed with plain Python formulas, for current and previous year. The formula is shown next to each ratio.
- In **Chat**, questions mentioning ratio, margin, ROE, turnover, days, etc. automatically get these pre-computed values as context, so the LLM quotes them instead of doing maths.
- Works for statements laid out like Indian Ind AS annual reports. Banks/NBFCs (different balance sheet format) are not supported yet.

## Page labels

Many annual reports are two-page spreads in one PDF page. Citations like `p.157R` mean PDF page 157, right half (`L` = left half).

## If you indexed with an older version

Delete the `chroma_db` folder, `pip install -r requirements.txt` again, restart Streamlit and re-index.

## Known MVP limits

- No OCR: scanned PDFs are flagged but not read.
- Small local models can slip on multi-step maths. Check the Sources panel.
- Hybrid search (vector + BM25) is in; no reranker yet.
- Indexing is slow on big reports (table detection): roughly 5-10 minutes for a 170-spread annual report, one time only.

## Next upgrades

1. Reranker
2. Multi-year / multi-company comparison
3. OCR for scanned PDFs (Qwen2.5-VL)
4. Cash flow statement metrics (CFO, free cash flow)
