# 🧠 AI Persona + Journey Builder

Turns **behavioral customer events** into **personas**, then generates **journey strategy** (channels, cadence, stage messaging) with an exportable brief.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What’s included
- Persona Overview (counts + feature sample)
- Journey Builder (stage strategy per persona)
- Exportable Journey Strategy Brief (Markdown + PDF if ReportLab is installed)

## Data
Files in `data/` are synthetic but modeled on Sephora product metadata and review patterns.
Swap in your real exports after identity stitching (customer_id / email hash).

## Extend next
- Add “what-if” simulator (channel/cadence changes → predicted impact)
- Add cohort KPIs by persona (repeat rate, churn risk, promo dependency)
- Add per-persona creative generator (subject lines, SMS copy, offers)


## What-If Simulator
Test channel, cadence, incentives, and personalization strength to estimate directional impact on engagement, conversion, fatigue, and opt-out risk.
