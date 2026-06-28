"""
Streamlit prototype application -- the demo the instructor will grade.

THREE TABS:
  1. Inspect   -> upload an image, see PASS/FAIL, score, and the defect heatmap.
  2. Dashboard -> live KPIs and the latest inspections, read from the database.
  3. About     -> which model is loaded and how to switch it.

It calls the SAME Orchestrator as the API, so what you demo is what you deploy.

RUN:  streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image

from src import config
from src.agents import Orchestrator
from src.database import db
from src.utils.images import overlay_heatmap

st.set_page_config(page_title="Industrial Vision Platform", page_icon="🔎", layout="wide")


# Build the orchestrator once and cache it (the model loads a single time).
@st.cache_resource
def get_orchestrator() -> Orchestrator:
    config.ensure_dirs()
    db.init_db()
    return Orchestrator(model_version=config.MODEL_BACKEND)


orch = get_orchestrator()

st.title("🔎 Industrial Vision Platform")
st.caption(f"Assembly-error detection · category **{config.CATEGORY}** · "
           f"model **{orch.inference.model_name}** · line **{config.LINE_ID}**")

tab_inspect, tab_dashboard, tab_about = st.tabs(["Inspect", "Dashboard", "About"])

# ---------------------------------------------------------------- Inspect tab
with tab_inspect:
    st.subheader("Inspect a part")
    uploaded = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg", "bmp"])
    part_id = st.text_input("Part ID (optional)", value="demo-part")

    if uploaded is not None:
        image = np.asarray(Image.open(uploaded).convert("RGB"), dtype=np.uint8)
        result = orch.inspect_one(part_id or "demo-part", image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Input", use_container_width=True)
        with col2:
            if result.get("heatmap") is not None:
                st.image(overlay_heatmap(image, result["heatmap"]),
                         caption="Anomaly heatmap (red = suspicious)",
                         use_container_width=True)
            else:
                st.info("This model does not produce a heatmap.")

        # Big, clear verdict banner.
        if result["verdict"] == "FAIL":
            st.error(f"❌ FAIL — {result['defect_type']} defect "
                     f"(severity: {result['severity']})")
        else:
            st.success("✅ PASS — no defect detected")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Anomaly score", f"{result['score']:.3f}")
        m2.metric("Threshold", f"{result['threshold']:.3f}")
        m3.metric("Confidence", f"{result['confidence']:.2f}")
        m4.metric("Latency", f"{result['latency_ms']} ms")
        st.caption(f"Saved to database as inspection #{result['inspection_id']} "
                   "(image archived for traceability).")

# -------------------------------------------------------------- Dashboard tab
with tab_dashboard:
    st.subheader("Live line analytics")
    stats = db.summary_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total inspected", stats["total"])
    c2.metric("Passed", stats["passed"])
    c3.metric("Failed", stats["failed"])
    c4.metric("Pass rate", f"{stats['pass_rate']*100:.1f}%")

    if stats["defect_breakdown"]:
        st.markdown("**Defects by type**")
        st.bar_chart(stats["defect_breakdown"])

    st.markdown("**Most recent inspections**")
    rows = db.recent_inspections(limit=25)
    if rows:
        table = [
            {
                "id": r["id"],
                "time": r["created_at"][11:19],
                "part": r["part_id"],
                "verdict": r["verdict"],
                "defect": r["defect_type"],
                "score": round(r["score"], 3),
                "latency_ms": r["latency_ms"],
            }
            for r in rows
        ]
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("No inspections yet — run some on the Inspect tab.")

# ------------------------------------------------------------------ About tab
with tab_about:
    st.subheader("About this system")
    st.markdown(
        f"""
This prototype is the **runnable core** of an industrial vision platform built as
six cooperating *agents* (acquisition → inference → decision → storage →
notification, wired by an orchestrator).

**Currently loaded model:** `{orch.inference.model_name}`

**Switch the model** without touching code — set an environment variable and restart:

```bash
export IVP_MODEL_BACKEND=autoencoder   # trained anomaly detector (recommended)
export IVP_MODEL_BACKEND=classifier    # trained ResNet18 good/defect classifier
export IVP_MODEL_BACKEND=dummy         # numpy baseline (no training needed)
```

**Switch the product category:**

```bash
export IVP_CATEGORY=juice_bottle       # or screw_bag, pushpins, breakfast_box, ...
```

See `docs/09_agents.md` for the full beginner→expert walkthrough of every agent.
"""
    )
