import streamlit as st
import pandas as pd
from pathlib import Path
from utils.persona_logic import build_features, assign_personas
from utils.journey_templates import default_persona_journeys
from utils.what_if_simulator import simulate_what_if

st.set_page_config(page_title="AI Persona + Journey Builder", layout="wide")
st.title("🧠 AI Persona + Journey Builder")
st.caption("Portfolio Demo • Personas from behavior → journey strategy → what-if planning → exportable brief")

DATA_DIR = Path("data")

@st.cache_data
def load_csvs():
    customers = pd.read_csv(DATA_DIR/"synthetic_customers.csv", parse_dates=["join_date"])
    events = pd.read_csv(DATA_DIR/"synthetic_customer_events.csv", parse_dates=["event_dt"])

    features_path = DATA_DIR/"customer_features.csv"
    persona_path = DATA_DIR/"persona_assignments.csv"
    journeys_path = DATA_DIR/"persona_journey_recommendations.csv"

    features = pd.read_csv(features_path) if features_path.exists() else None
    personas = pd.read_csv(persona_path) if persona_path.exists() else None
    journeys = pd.read_csv(journeys_path) if journeys_path.exists() else default_persona_journeys()
    return customers, events, features, personas, journeys

customers, events, features_pre, personas_pre, journeys = load_csvs()

# Sidebar controls
st.sidebar.header("Controls")
mode = st.sidebar.radio("Mode", ["Use precomputed personas", "Recompute personas from events"], index=0)
persona_filter = st.sidebar.multiselect("Persona filter", sorted(journeys["persona"].unique().tolist()))
tier_filter = st.sidebar.multiselect("Loyalty tier", sorted(customers["loyalty_tier"].unique().tolist()))
channel_filter = st.sidebar.multiselect("Preferred channel", sorted(customers["pref_channel"].unique().tolist()))

# Build/Load features & personas
if mode == "Recompute personas from events" or personas_pre is None or features_pre is None:
    features = build_features(customers, events)
    personas = assign_personas(features, customers)
else:
    features = features_pre
    personas = personas_pre

# Merge for display
df = customers.merge(features, on="customer_id", how="left").merge(personas, on="customer_id", how="left")

if tier_filter:
    df = df[df["loyalty_tier"].isin(tier_filter)]
if channel_filter:
    df = df[df["pref_channel"].isin(channel_filter)]
if persona_filter:
    df = df[df["persona"].isin(persona_filter)]

# Topline metrics (for the filtered view)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(df):,}")
c2.metric("Avg spend (90d)", f"${df['spend_90'].mean():.2f}")
c3.metric("Avg orders (90d)", f"{df['purchase_count_90'].mean():.2f}")
c4.metric("Avg recency (days)", f"{df['recency_days'].mean():.0f}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Persona Overview", "Journey Builder", "What-If Simulator", "Export Brief"])

with tab1:
    st.subheader("Persona Overview")
    counts = df["persona"].value_counts().reset_index()
    counts.columns = ["persona", "count"]
    st.dataframe(counts, use_container_width=True)

    with st.expander("View customer feature sample"):
        st.dataframe(df.sample(min(300, len(df))), use_container_width=True, height=360)

with tab2:
    st.subheader("Journey Builder")
    persona_selected = st.selectbox("Select persona", sorted(journeys["persona"].unique().tolist()), key="journey_persona")
    journey = journeys[journeys["persona"] == persona_selected].iloc[0].to_dict()

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"### {persona_selected}")
        st.write(f"**Preferred channels:** {journey.get('preferred_channels','')}")
        st.write(f"**Cadence guidance:** {journey.get('cadence_guidance','')}")
        st.write(f"**Core motivation:** {journey.get('core_motivation','')}")
        st.write(f"**Primary barrier:** {journey.get('primary_barrier','')}")
    with right:
        st.markdown("### Stage Strategy")
        stage_rows = pd.DataFrame([
            {"stage": "Awareness", "strategy": journey.get("awareness", "")},
            {"stage": "Consideration", "strategy": journey.get("consideration", "")},
            {"stage": "Conversion", "strategy": journey.get("conversion", "")},
            {"stage": "Retention", "strategy": journey.get("retention", "")},
            {"stage": "Advocacy", "strategy": journey.get("advocacy", "")},
        ])
        st.dataframe(stage_rows, use_container_width=True, height=260)

with tab3:
    st.subheader("What-If Simulator")
    st.caption("Test channel + cadence + incentives + personalization and see the predicted directional impact.")

    persona_sim = st.selectbox("Persona", sorted(journeys["persona"].unique().tolist()), key="sim_persona")
    seg = df[df["persona"] == persona_sim].copy()

    colA, colB = st.columns([1, 1])
    with colA:
        current_channel = st.selectbox("Current primary channel", ["email", "sms", "app_push"], index=0)
        new_channel = st.selectbox("What-if channel", ["email", "sms", "app_push"], index=1)
        touches_per_week = st.slider("Touches per week", min_value=0.5, max_value=6.0, value=2.0, step=0.5)
    with colB:
        incentive_level = st.slider("Incentive level (none → strong)", 0.0, 1.0, 0.4, 0.05)
        ai_personalization = st.slider("AI personalization strength", 0.0, 1.0, 0.6, 0.05)

    result = simulate_what_if(
        segment_df=seg,
        current_channel=current_channel,
        new_channel=new_channel,
        touches_per_week=float(touches_per_week),
        incentive_level=float(incentive_level),
        ai_personalization=float(ai_personalization),
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Engagement Index", f"{result['engagement_index']*100:.1f}")
    m2.metric("Conversion Probability", f"{result['conversion_prob']*100:.1f}%")
    m3.metric("Fatigue Risk", f"{result['fatigue_risk']*100:.1f}%")
    m4.metric("Unsub Risk", f"{result['unsub_risk']*100:.1f}%")

    st.markdown("#### Notes")
    for n in result["notes"]:
        st.write(f"- {n}")

    with st.expander("Segment snapshot used for simulation"):
        st.write(f"Customers in persona segment: **{len(seg):,}**")
        snap = pd.DataFrame([{
            "avg_spend_90": seg["spend_90"].mean() if len(seg) else 0,
            "avg_orders_90": seg["purchase_count_90"].mean() if len(seg) else 0,
            "avg_discount_share_90": seg["discount_share_90"].mean() if len(seg) else 0,
            "avg_premium_share_90": seg["premium_share_90"].mean() if len(seg) else 0,
            "avg_recency_days": seg["recency_days"].mean() if len(seg) else 0,
        }])
        st.dataframe(snap, use_container_width=True)

with tab4:
    st.subheader("Export Journey Strategy Brief")
    persona_selected = st.selectbox("Select persona", sorted(journeys["persona"].unique().tolist()), key="export_persona")
    journey = journeys[journeys["persona"] == persona_selected].iloc[0].to_dict()
    segment = df[df["persona"] == persona_selected].copy()

    def build_brief_md(persona_name: str, j: dict, segment_df: pd.DataFrame) -> str:
        stats = {
            "customers": int(len(segment_df)),
            "avg_spend_90": float(segment_df["spend_90"].mean() if len(segment_df) else 0),
            "avg_orders_90": float(segment_df["purchase_count_90"].mean() if len(segment_df) else 0),
            "avg_discount_share_90": float(segment_df["discount_share_90"].mean() if len(segment_df) else 0),
            "avg_premium_share_90": float(segment_df["premium_share_90"].mean() if len(segment_df) else 0),
        }
        lines = []
        lines.append(f"# Journey Strategy Brief — {persona_name}\n")
        lines.append("## Persona Summary")
        lines.append(f"- Preferred channels: {j.get('preferred_channels','')}")
        lines.append(f"- Cadence guidance: {j.get('cadence_guidance','')}")
        lines.append(f"- Core motivation: {j.get('core_motivation','')}")
        lines.append(f"- Primary barrier: {j.get('primary_barrier','')}\n")
        lines.append("## Segment Snapshot")
        lines.append(f"- Customers in segment: {stats['customers']:,}")
        lines.append(f"- Avg spend (90d): ${stats['avg_spend_90']:.2f}")
        lines.append(f"- Avg orders (90d): {stats['avg_orders_90']:.2f}")
        lines.append(f"- Avg discount share (90d): {stats['avg_discount_share_90']*100:.1f}%")
        lines.append(f"- Avg premium share (90d): {stats['avg_premium_share_90']*100:.1f}%\n")
        lines.append("## Stage Strategy")
        lines.append(f"- Awareness: {j.get('awareness','')}")
        lines.append(f"- Consideration: {j.get('consideration','')}")
        lines.append(f"- Conversion: {j.get('conversion','')}")
        lines.append(f"- Retention: {j.get('retention','')}")
        lines.append(f"- Advocacy: {j.get('advocacy','')}\n")
        lines.append("## KPIs to Watch")
        lines.append("- CTR/CTOR by channel, repeat purchase rate, churn risk, promo dependency, opt-out rate, time-to-next-purchase\n")
        lines.append("## Compliance Notes")
        lines.append("- Respect opt-in per channel; use frequency caps to prevent fatigue; surface unsubscribe trends.\n")
        return "\n".join(lines)

    brief_md = build_brief_md(persona_selected, journey, segment)
    st.code(brief_md, language="markdown")
    st.download_button("Download Brief (Markdown)", data=brief_md, file_name=f"journey_brief_{persona_selected.lower().replace(' ','_').replace('/','_')}.md", mime="text/markdown")

    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.pagesizes import LETTER
        import io

        def md_to_pdf_bytes(md: str) -> bytes:
            styles = getSampleStyleSheet()
            story = []
            for line in md.split("\n"):
                if line.startswith("# "):
                    story.append(Paragraph(f"<b>{line[2:]}</b>", styles["Title"]))
                elif line.startswith("## "):
                    story.append(Spacer(1, 8))
                    story.append(Paragraph(f"<b>{line[3:]}</b>", styles["Heading2"]))
                elif line.strip().startswith("- "):
                    story.append(Paragraph(line, styles["Normal"]))
                else:
                    if line.strip():
                        story.append(Paragraph(line, styles["Normal"]))
                story.append(Spacer(1, 6))
            buff = io.BytesIO()
            doc = SimpleDocTemplate(buff, pagesize=LETTER)
            doc.build(story)
            return buff.getvalue()

        pdf_bytes = md_to_pdf_bytes(brief_md)
        st.download_button("Download Brief (PDF)", data=pdf_bytes, file_name=f"journey_brief_{persona_selected.lower().replace(' ','_').replace('/','_')}.pdf", mime="application/pdf")
    except Exception:
        st.info("PDF export unavailable here; Markdown export is ready.")

st.divider()
st.caption("Tip: Swap synthetic CSVs with real customer events once you add identity stitching (customer_id / email hash).")
