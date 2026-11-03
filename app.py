import os
import tempfile

import pandas as pd
import streamlit as st

import rag
import ratios

st.set_page_config(page_title="Financial Doc Analyst", page_icon="📊", layout="wide")
st.title("📊 Financial Document Analyst")
st.caption("Local RAG: LM Studio + ChromaDB. Answers come only from your PDFs, with page citations.")

models = rag.chat_models()

with st.sidebar:
    st.header("1. Documents")
    files = st.file_uploader("Upload annual reports, statements, audit reports", type="pdf", accept_multiple_files=True)
    if st.button("Index uploaded PDFs", type="primary", disabled=not files):
        for f in files:
            with st.spinner(f"Indexing {f.name} ..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                try:
                    try:  # statements -> ratio calculator (fast, no LLM)
                        for basis, d in ratios.extract_pdf(tmp.name).items():
                            ratios.save(f.name, basis, d)
                    except Exception as e:
                        st.warning(f"{f.name}: ratio extraction skipped ({e})")
                    n, scanned = rag.ingest_pdf(tmp.name, f.name)
                    if scanned:
                        st.warning(f"{f.name}: little or no text found. Scanned PDF? OCR is not in this MVP.")
                    else:
                        st.success(f"{f.name}: {n} chunks indexed")
                except Exception as e:
                    st.error(f"{f.name}: {e}")
                finally:
                    os.unlink(tmp.name)

    docs = rag.list_docs()
    st.header("2. Scope")
    selected = st.multiselect("Search in (empty = all)", docs)
    if docs:
        to_delete = st.selectbox("Remove a document", [""] + docs)
        if to_delete and st.button("Delete"):
            rag.delete_doc(to_delete)
            ratios.delete(to_delete)
            st.rerun()

    st.header("3. Model")
    if models:
        model = st.selectbox("Chat model", models)
    else:
        model = None
        st.error("LM Studio server not reachable or no chat model loaded. Start the server in the Developer tab and load a model.")
    top_k = st.slider("Chunks to retrieve", 3, 10, 6)

view = st.radio("View", ["💬 Chat", "📈 Ratios"], horizontal=True, label_visibility="collapsed")

# ------------------------------------------------------------------ Ratios view
if view == "📈 Ratios":
    with st.expander("Extract statements from a PDF (fast, no indexing needed)"):
        rf = st.file_uploader("Annual report PDF", type="pdf", key="ratio_pdf")
        if rf and st.button("Extract balance sheet + P&L"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(rf.getvalue())
            try:
                found = ratios.extract_pdf(tmp.name)
                if not found:
                    st.error("Could not find a balance sheet and P&L in this PDF.")
                for basis, d in found.items():
                    ratios.save(rf.name, basis, d)
                    st.success(f"{rf.name}: {basis} statements extracted ({len(d['metrics'])} line items)")
            finally:
                os.unlink(tmp.name)

    saved = ratios.list_docs()
    if not saved:
        st.info("No statements yet. Index a PDF from the sidebar or use the extractor above.")
        st.stop()

    doc = st.selectbox("Document", saved)
    basis = st.radio("Statements", ratios.bases_for(doc), horizontal=True)
    data = ratios.load(doc, basis)
    st.caption(f"{data['units']}  |  Current = {data['cur_label']}, Previous = {data['prev_label']}")

    st.subheader("1. Extracted line items (edit if anything looks wrong)")
    rows = []
    for k, label in ratios.LABELS.items():
        m = data["metrics"].get(k, {})
        rows.append({"key": k, "Line item": label, "Current": m.get("cur"), "Previous": m.get("prev"),
                     "Source page": m.get("page", "not found")})
    edited = st.data_editor(
        pd.DataFrame(rows), hide_index=True, height=420,
        disabled=["key", "Line item", "Source page"], column_config={"key": None},
        key=f"editor-{doc}-{basis}",
    )
    metrics = {}
    for _, r in edited.iterrows():
        old = data["metrics"].get(r["key"], {})
        cur = None if pd.isna(r["Current"]) else float(r["Current"])
        prev = None if pd.isna(r["Previous"]) else float(r["Previous"])
        if cur is not None or prev is not None:
            metrics[r["key"]] = {"cur": cur, "prev": prev, "page": old.get("page", "edited")}
    if st.button("Save edits"):
        data["metrics"] = metrics
        ratios.save(doc, basis, data)
        st.success("Saved. Chat will use these numbers.")

    st.subheader("2. Sanity checks")
    for period, label in (("cur", data["cur_label"]), ("prev", data["prev_label"])):
        for desc, ok, detail in ratios.sanity_checks(metrics, period):
            (st.success if ok else st.error)(f"{label}: {desc} ({detail})")

    st.subheader("3. Ratios")
    cur = ratios.compute_ratios(ratios._period_values(metrics, "cur"))
    prev = ratios.compute_ratios(ratios._period_values(metrics, "prev"))
    table = [{"Group": g, "Ratio": n, "Formula": f, data["cur_label"]: ratios.fmt(a, k),
              data["prev_label"]: ratios.fmt(b, k)} for (g, n, f, a, k), (_, _, _, b, _) in zip(cur, prev)]
    st.dataframe(pd.DataFrame(table), hide_index=True)
    st.caption("All ratios are computed in Python from the numbers above, not by the LLM. "
               "COGS = materials consumed + purchases + changes in inventories (suits manufacturers).")
    st.stop()

# ------------------------------------------------------------------ Chat view
if "history" not in st.session_state:
    st.session_state.history = []

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("used_ratios"):
            st.caption("Used pre-computed ratios from the statements.")
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"**{s['doc']}, p.{s['page']}** ({s['kind']})")
                    st.code(s["text"][:1500], language="markdown")

if q := st.chat_input("Ask e.g. 'What were the key audit matters?' or 'What is the current ratio?'"):
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        if not model:
            st.error("No chat model available.")
        elif not docs:
            st.info("Upload and index at least one PDF first.")
        else:
            ctx = rag.retrieve(q, selected or None, k=top_k)
            extra = ""
            if ratios.is_ratio_question(q):
                targets = [d for d in (selected or docs) if d in ratios.list_docs()][:3]
                extra = ratios.context_block(targets)
            if extra:
                st.caption("Using pre-computed ratios from the statements.")
            reply = st.write_stream(rag.answer_stream(q, ctx, model, extra=extra))
            with st.expander("Sources"):
                for s in ctx:
                    st.markdown(f"**{s['doc']}, p.{s['page']}** ({s['kind']})")
                    st.code(s["text"][:1500], language="markdown")
            st.session_state.history.append(
                {"role": "assistant", "content": reply, "sources": ctx, "used_ratios": bool(extra)})
