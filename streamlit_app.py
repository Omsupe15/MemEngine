"""Streamlit Web UI for MemEngine.

Interactive local-first memory console for AI chatbots:
- Ingest conversation turns with small-talk filtering and conflict detection
- Recall active memories via hybrid semantic search with rejection of unknowns
- Stored Memory Ledger showing historical lineage (active vs. superseded)
- Interactive 1-click Canonical Test runner
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import Optional

from app.config import get_settings
from app.retrieval import get_recall_engine
from app.schemas import IngestRequest, RecallRequest
from app.storage import get_storage_manager
from app.updates import get_conflict_resolver
from app.evaluator import EvaluationSummary, run_full_evaluation
from scripts.benchmark import run_latency_benchmark

# Page Configuration
st.set_page_config(
    page_title="MemEngine - Local Memory Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Dark Modern Aesthetic
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }
    .badge-active { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-superseded { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-type { background-color: rgba(99, 102, 241, 0.2); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.4); }
    .lineage-text {
        font-size: 0.82rem;
        color: #f59e0b;
        font-weight: 500;
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_services():
    """Retrieve backend singleton services."""
    storage = get_storage_manager()
    resolver = get_conflict_resolver()
    recall_engine = get_recall_engine()
    return storage, resolver, recall_engine


storage, resolver, recall_engine = get_services()

# --- Sidebar Controls & Metrics ---
with st.sidebar:
    st.markdown("### 🧠 Engine Telemetry")
    active_mems = storage.get_active_memories()
    total_count = storage.sqlite.count_memories()
    superseded_count = storage.sqlite.count_memories(status="superseded")

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Active", len(active_mems))
    col_m2.metric("Superseded", superseded_count)
    st.metric("Total Stored", total_count)

    st.divider()
    st.markdown("### ⚙️ Retrieval Knobs")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5, step=1)
    threshold = st.slider(
        "Relevance Cutoff Threshold",
        min_value=0.30,
        max_value=0.90,
        value=0.52,
        step=0.02,
        help="Scores below this cutoff are rejected as unknown.",
    )

    st.divider()
    st.markdown("### ⚡ Quick Demo")
    if st.button("🚀 Load 4 Canonical Messages", use_container_width=True):
        resolver.process_ingest_request(IngestRequest(message="I use Arch Linux."))
        resolver.process_ingest_request(IngestRequest(message="I'm building a Next.js app."))
        resolver.process_ingest_request(IngestRequest(message="My project deadline is October 15."))
        resolver.process_ingest_request(IngestRequest(message="Actually, the deadline moved to November 2."))
        st.success("Loaded Arch Linux, Next.js, and November 2 deadline!")
        st.rerun()

    if st.button("🔄 Refresh View", use_container_width=True):
        st.rerun()


# --- Main Dashboard Header ---
st.markdown('<div class="main-title">MemEngine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Local-first cognitive memory engine for AI chatbots • Zero paid APIs • Offline CPU embeddings</div>',
    unsafe_allow_html=True,
)

# Navigation Tabs
tab_recall, tab_remember, tab_ledger, tab_workflow, tab_evaluation = st.tabs([
    "🔍 Recall & Semantic Search",
    "📥 Ingest & Conflict Resolution",
    "📚 Memory Ledger & Lineage",
    "🧪 Automated Canonical Test",
    "📊 Benchmark & Evaluation Report",
])

# -------------------------------------------------------------
# TAB 1: RECALL (Search & Strict Unknown Rejection)
# -------------------------------------------------------------
with tab_recall:
    st.subheader("Query Stored Memories")
    st.caption("Searches active memories using hybrid semantic vector similarity (ChromaDB) + BM25Okapi lexical matching.")

    # Quick test buttons
    st.markdown("**Canonical Sample Queries:**")
    cq_cols = st.columns(3)
    preset_query: Optional[str] = None
    if cq_cols[0].button("🐧 'What OS does the user use?'", use_container_width=True):
        preset_query = "What OS does the user use?"
    if cq_cols[1].button("📅 'When is the project deadline?'", use_container_width=True):
        preset_query = "When is the project deadline?"
    if cq_cols[2].button("🍕 'What is the user's favourite food?' (Negative)", use_container_width=True):
        preset_query = "What is the user's favourite food?"

    query_input = st.text_input(
        "Enter query for memory recall:",
        value=preset_query or "",
        placeholder="e.g. When is the project deadline?",
        key="recall_query_input",
    )

    if query_input:
        req = RecallRequest(query=query_input, top_k=top_k, threshold=threshold)
        response = recall_engine.recall(req)

        if response.memories:
            st.success(f"Found {len(response.memories)} relevant active memory(s):")
            for idx, item in enumerate(response.memories, 1):
                score_pct = int(round(item.score * 100))
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(
                            f'<span class="badge-pill badge-type">{item.type}</span>'
                            f'<span class="badge-pill badge-active">Match {score_pct}%</span>'
                            f'<code>#{item.id}</code>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"### {item.text}")
                        st.markdown(
                            f"**Entity Key:** `{item.subject}` &rarr; `{item.attribute}` = **{item.value}**"
                        )
                    with c2:
                        st.progress(item.score, text=f"Score: {item.score:.3f}")
        else:
            st.warning(
                f"⚠️ **No relevant memory found.** All active candidates scored below threshold `{threshold:.2f}`."
            )
            st.info(
                "✅ **Calibrated Rejection**: Engine correctly returned an empty list `[]` instead of hallucinating unrelated memories."
            )

# -------------------------------------------------------------
# TAB 2: INGEST (Remember, Small Talk Filter, Conflict Detection)
# -------------------------------------------------------------
with tab_remember:
    st.subheader("Ingest Conversation Turn (Remember)")
    st.caption("Sends a message to the engine. Filters out small talk, extracts structured facts, and detects updates/contradictions.")

    st.markdown("**Try Canonical Messages:**")
    cm_cols = st.columns(5)
    preset_msg: Optional[str] = None
    if cm_cols[0].button("🐧 Arch Linux", use_container_width=True):
        preset_msg = "I use Arch Linux."
    if cm_cols[1].button("⚡ Next.js", use_container_width=True):
        preset_msg = "I'm building a Next.js app."
    if cm_cols[2].button("📅 Oct 15 Deadline", use_container_width=True):
        preset_msg = "My project deadline is October 15."
    if cm_cols[3].button("🔄 Nov 2 Update", use_container_width=True):
        preset_msg = "Actually, the deadline moved to November 2."
    if cm_cols[4].button("💬 'ok thanks'", use_container_width=True):
        preset_msg = "ok thanks"

    with st.form("remember_form", clear_on_submit=True):
        msg_text = st.text_input(
            "User message to ingest:",
            value=preset_msg or "",
            placeholder="e.g. Actually, the deadline moved to November 2.",
        )
        submitted = st.form_submit_button("Remember Message", use_container_width=True)

        if submitted and msg_text.strip():
            req = IngestRequest(message=msg_text.strip())
            res = resolver.process_ingest_request(req)

            if res.status == "created":
                st.success(f"✅ **Memory Created (#{res.memory.id})**: {res.message}")
                with st.expander("Inspected Memory Record", expanded=True):
                    st.json(res.memory.model_dump())
            elif res.status == "superseded":
                st.warning(f"🔄 **Memory Updated (#{res.memory.id})**: {res.message}")
                st.info(f"Historical Lineage: Superseded Memory #{res.superseded_id} and marked it inactive.")
                with st.expander("Inspected Memory Record", expanded=True):
                    st.json(res.memory.model_dump())
            elif res.status == "ignored":
                st.info(f"💬 **Small Talk Filtered**: {res.message}")
            elif res.status == "reasserted":
                st.info(f"ℹ️ **Idempotent Fact**: {res.message}")

# -------------------------------------------------------------
# TAB 3: LEDGER (Historical Audit & Lineage)
# -------------------------------------------------------------
with tab_ledger:
    st.subheader("Stored Memories Ledger")
    st.caption("Inspect all facts stored in SQLite & ChromaDB with historical lineage.")

    filter_option = st.radio(
        "Filter by Status:",
        ["all", "active", "superseded", "deleted"],
        horizontal=True,
    )

    records = storage.list_memories(
        status=None if filter_option == "all" else filter_option,
        limit=200,
    )

    if records:
        for mem in records:
            with st.container(border=True):
                c_head, c_btn = st.columns([4, 1])
                with c_head:
                    status_class = "badge-active" if mem.status == "active" else "badge-superseded"
                    st.markdown(
                        f'<span class="badge-pill {status_class}">{mem.status}</span>'
                        f'<span class="badge-pill badge-type">{mem.type}</span>'
                        f'<code>#{mem.id}</code>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{mem.text}**")
                    st.caption(
                        f"Subject: `{mem.subject}` | Attribute: `{mem.attribute}` | Value: `{mem.value}` | Importance: `{mem.importance}`"
                    )
                    if mem.replaced_by:
                        st.markdown(
                            f'<div class="lineage-text">↪ Replaced by Memory #{mem.replaced_by}</div>',
                            unsafe_allow_html=True,
                        )
                with c_btn:
                    if mem.status != "deleted":
                        if st.button("🗑️ Delete", key=f"del_{mem.id}"):
                            storage.delete_memory(mem.id)
                            st.rerun()
    else:
        st.info("No memories found for this filter.")

# -------------------------------------------------------------
# TAB 4: AUTOMATED CANONICAL TEST RUNNER
# -------------------------------------------------------------
with tab_workflow:
    st.subheader("Run Full Canonical User Workflow")
    st.markdown(
        """
        Executes the exact 4-message user flow from the specification:
        1. `remember("I use Arch Linux.")`
        2. `remember("I'm building a Next.js app.")`
        3. `remember("My project deadline is October 15.")`
        4. `remember("Actually, the deadline moved to November 2.")`
        
        Then queries the engine:
        - `recall("What OS does the user use?")` $\\to$ **"User uses Arch Linux."**
        - `recall("When is the project deadline?")` $\\to$ **"Deadline is November 2."** *(October 15 excluded)*
        - `recall("What is the user's favourite food?")` $\\to$ **[]** *(rejected without hallucination)*
        """
    )

    if st.button("▶️ Execute Canonical Benchmark Now", type="primary", use_container_width=True):
        results_container = st.container(border=True)
        with results_container:
            st.write("### Step 1: Ingesting Messages")
            
            r1 = resolver.process_ingest_request(IngestRequest(message="I use Arch Linux."))
            st.write(f"1. Ingest Arch Linux -> Status: **{r1.status}** (`#{r1.memory.id}`)")
            
            r2 = resolver.process_ingest_request(IngestRequest(message="I'm building a Next.js app."))
            st.write(f"2. Ingest Next.js -> Status: **{r2.status}** (`#{r2.memory.id}`)")
            
            r3 = resolver.process_ingest_request(IngestRequest(message="My project deadline is October 15."))
            id_oct = r3.memory.id
            st.write(f"3. Ingest Oct 15 -> Status: **{r3.status}** (`#{id_oct}`)")
            
            r4 = resolver.process_ingest_request(IngestRequest(message="Actually, the deadline moved to November 2."))
            id_nov = r4.memory.id
            st.write(f"4. Ingest Nov 2 Update -> Status: **{r4.status}** (Replaced `#{r4.superseded_id}` with `#{id_nov}`)")
            
            st.divider()
            st.write("### Step 2: Verification of Recalls")
            
            q1 = recall_engine.recall(RecallRequest(query="What OS does the user use?"))
            pass_q1 = len(q1.memories) > 0 and "Arch Linux" in q1.memories[0].text
            st.write(f"• Query: *'What OS does the user use?'* -> {'✅ PASS' if pass_q1 else '❌ FAIL'}")
            if pass_q1:
                st.caption(f"  Returned: `{q1.memories[0].text}` (Score: {q1.memories[0].score:.3f})")
            
            q2 = recall_engine.recall(RecallRequest(query="When is the project deadline?"))
            pass_q2 = (
                len(q2.memories) > 0
                and "November 2" in q2.memories[0].text
                and not any("October 15" in m.text for m in q2.memories)
            )
            st.write(f"• Query: *'When is the project deadline?'* -> {'✅ PASS' if pass_q2 else '❌ FAIL'}")
            if pass_q2:
                st.caption(f"  Returned: `{q2.memories[0].text}` (Score: {q2.memories[0].score:.3f}) | Oct 15 excluded: YES")
            
            q3 = recall_engine.recall(RecallRequest(query="What is the user's favourite food?"))
            pass_q3 = len(q3.memories) == 0
            st.write(f"• Query: *'What is the user's favourite food?'* -> {'✅ PASS' if pass_q3 else '❌ FAIL'}")
            if pass_q3:
                st.caption("  Correctly returned empty list `[]` (Rejection without hallucination)")
            
            if pass_q1 and pass_q2 and pass_q3:
                st.balloons()
                st.success("🎉 All canonical tests passed with 100% compliance!")

# -------------------------------------------------------------
# TAB 5: AUTOMATED BENCHMARK & EVALUATION REPORT
# -------------------------------------------------------------
with tab_evaluation:
    st.subheader("📊 Comprehensive Benchmark Suite & Evaluation Report")
    st.caption("Evaluates 35 categorized test cases covering Direct Facts, Preferences, Project Details, Updates, Contradictions, Multi-relevant queries, and Unknown Rejection.")

    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        run_bench = st.button("▶️ Run Full 35-Case Benchmark Suite", type="primary", use_container_width=True)
    with col_b2:
        run_lat = st.button("⚡ Profile Latency (p50 / p95 / p99)", use_container_width=True)

    # Run Benchmark if triggered
    if run_bench:
        progress_bar = st.progress(0, text="Initializing evaluation harness...")
        status_text = st.empty()

        def update_progress(idx, total, res):
            progress_bar.progress(idx / total, text=f"Evaluating [{idx}/{total}]: {res.case_id} ({res.category})")
            status_text.caption(f"Last test: **{res.case_id}** &rarr; {'✅ PASS' if res.passed else '❌ FAIL'}: {res.query}")

        summary = run_full_evaluation(progress_callback=update_progress)
        progress_bar.progress(1.0, text="Benchmark complete!")
        status_text.empty()
        st.session_state["eval_summary"] = summary
        st.success(f"🎉 Benchmark complete! Overall Accuracy: **{summary.overall_accuracy:.1f}%** ({summary.passed_cases}/{summary.total_cases} cases passed)")

    # Display Benchmark Summary if present in session state
    if "eval_summary" in st.session_state:
        summary: EvaluationSummary = st.session_state["eval_summary"]

        st.markdown("### 🏆 Overall Benchmark Metrics")
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
        m_col1.metric("Overall Accuracy", f"{summary.overall_accuracy:.1f}%")
        m_col2.metric("Recall@5", f"{summary.recall_at_5:.1f}%")
        m_col3.metric("Top-1 Accuracy", f"{summary.top_1_accuracy:.1f}%")
        m_col4.metric("Unknown Rejection", f"{summary.unknown_rejection_rate:.1f}%")
        m_col5.metric("Supersede Integrity", f"{summary.supersede_integrity_rate:.1f}%")
        m_col6.metric("Avg Latency", f"{summary.avg_latency_ms:.1f} ms")

        st.divider()

        # Category Breakdown
        st.markdown("### 📂 Performance by Category")
        c_left, c_right = st.columns([3, 2])

        cat_rows = []
        for cat, stats in summary.category_metrics.items():
            cat_rows.append({
                "Category": cat,
                "Passed": stats["passed"],
                "Total": stats["total"],
                "Accuracy (%)": stats["accuracy"],
            })
        df_cats = pd.DataFrame(cat_rows)

        with c_left:
            st.dataframe(df_cats, use_container_width=True, hide_index=True)
        with c_right:
            st.bar_chart(df_cats.set_index("Category")["Accuracy (%)"], color="#6366f1")

        st.divider()

        # Case-by-case Interactive Diagnostic Inspector
        st.markdown("### 🔍 Case-by-Case Diagnostic Inspector")
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            all_categories = ["All"] + list(summary.category_metrics.keys())
            selected_cat = st.selectbox("Filter Category:", all_categories)
        with filter_col2:
            status_filter = st.selectbox("Filter Status:", ["All", "Passed Only", "Failed Only"])

        filtered_cases = summary.case_results
        if selected_cat != "All":
            filtered_cases = [c for c in filtered_cases if c.category == selected_cat]
        if status_filter == "Passed Only":
            filtered_cases = [c for c in filtered_cases if c.passed]
        elif status_filter == "Failed Only":
            filtered_cases = [c for c in filtered_cases if not c.passed]

        st.caption(f"Showing {len(filtered_cases)} of {summary.total_cases} test cases:")

        for case_res in filtered_cases:
            badge_color = "badge-active" if case_res.passed else "badge-superseded"
            badge_text = "PASS" if case_res.passed else "FAIL"
            with st.container(border=True):
                h1, h2 = st.columns([4, 1])
                with h1:
                    st.markdown(
                        f'<span class="badge-pill {badge_color}">{badge_text}</span>'
                        f'<span class="badge-pill badge-type">{case_res.category}</span>'
                        f'<code>{case_res.case_id}</code>: **{case_res.description}**',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Query:** *\"{case_res.query}\"*")
                    st.caption(f"Retrieved: {case_res.retrieved_count} memories | Top score: {case_res.top_score or 0.0:.3f} | Latency: {case_res.latency_ms}ms")
                    if case_res.top_text:
                        st.markdown(f"**Top Result:** `{case_res.top_text}`")
                    st.info(f"Verdict: {case_res.notes}")
                with h2:
                    if case_res.passed:
                        st.success("Verified ✅")
                    else:
                        st.error("Needs Review ❌")

    # Run Latency Benchmark if triggered
    if run_lat:
        with st.spinner("Measuring p50, p95, and p99 latency across 100 sample operations..."):
            ing_p, rec_p = run_latency_benchmark(n_samples=100)
            st.session_state["lat_profiles"] = (ing_p, rec_p)

    if "lat_profiles" in st.session_state:
        ing_p, rec_p = st.session_state["lat_profiles"]
        st.markdown("### ⚡ Latency & Throughput Benchmark (100 Samples)")

        lat_col1, lat_col2 = st.columns(2)
        with lat_col1:
            with st.container(border=True):
                st.markdown("#### 📥 Ingestion & Conflict Detection")
                p1, p2, p3 = st.columns(3)
                p1.metric("p50 Latency", f"{ing_p.p50_ms:.2f} ms")
                p2.metric("p95 Latency", f"{ing_p.p95_ms:.2f} ms")
                p3.metric("p99 Latency", f"{ing_p.p99_ms:.2f} ms")
                st.caption(f"Mean: {ing_p.mean_ms:.2f} ms | Throughput: **{ing_p.throughput_ops_sec:.1f} ops/sec**")

        with lat_col2:
            with st.container(border=True):
                st.markdown("#### 🔍 Hybrid Semantic Recall")
                r1, r2, r3 = st.columns(3)
                r1.metric("p50 Latency", f"{rec_p.p50_ms:.2f} ms")
                r2.metric("p95 Latency", f"{rec_p.p95_ms:.2f} ms")
                r3.metric("p99 Latency", f"{rec_p.p99_ms:.2f} ms")
                st.caption(f"Mean: {rec_p.mean_ms:.2f} ms | Throughput: **{rec_p.throughput_ops_sec:.1f} ops/sec**")

