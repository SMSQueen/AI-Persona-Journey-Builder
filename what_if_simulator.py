import numpy as np
import pandas as pd

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def simulate_what_if(
    segment_df: pd.DataFrame,
    current_channel: str,
    new_channel: str,
    touches_per_week: float,
    incentive_level: float,
    ai_personalization: float,
) -> dict:
    """
    Lightweight simulator to estimate directional impact on:
    - engagement (open/click proxy)
    - conversion probability (proxy)
    - fatigue risk
    - unsubscribe risk

    Inputs are intentionally simple so the model is explainable in interviews.
    * incentive_level: 0..1 (none -> strong)
    * ai_personalization: 0..1 (basic rules -> highly personalized)
    """
    if segment_df is None or len(segment_df) == 0:
        return {
            "engagement_index": 0.0,
            "conversion_prob": 0.0,
            "fatigue_risk": 0.0,
            "unsub_risk": 0.0,
            "notes": ["No customers in segment."]
        }

    # Baselines derived from segment behavior
    spend_norm = segment_df["spend_90"].mean() / (segment_df["spend_90"].quantile(0.95) + 1e-9)
    freq_norm = segment_df["purchase_count_90"].mean() / (segment_df["purchase_count_90"].quantile(0.95) + 1e-9)
    disc_norm = segment_df["discount_share_90"].mean()
    prem_norm = segment_df["premium_share_90"].mean()
    recency = segment_df["recency_days"].mean()

    # Base engagement index (0..1)
    base_engagement = clamp01(0.25 + 0.35*freq_norm + 0.20*spend_norm + 0.10*(1 - disc_norm) + 0.10*prem_norm)

    # Channel multipliers (heuristic; explainable)
    # Assumption: sms/app_push drive faster response; email better for depth/education.
    channel_mult = {
        "email": 1.00,
        "sms": 1.15,
        "app_push": 1.10
    }
    cur_mult = channel_mult.get(current_channel, 1.0)
    new_mult = channel_mult.get(new_channel, 1.0)

    # Touch cadence effect: increasing touches can lift engagement up to a point, then fatigue hits
    # Soft cap around 2.5 touches/week
    cadence_lift = clamp01(0.85 + 0.12*min(touches_per_week, 2.5))
    fatigue_penalty = clamp01(1.0 - 0.10*max(0.0, touches_per_week - 2.5))

    # Incentive helps more for deal-driven segments; less for premium segments
    incentive_boost = clamp01(0.05 + 0.25*incentive_level*(0.7*disc_norm + 0.3*(1-prem_norm)))

    # Personalization helps more when recency is lower and premium share is higher (luxury expects relevance)
    personalization_boost = clamp01(0.04 + 0.22*ai_personalization*(0.6*prem_norm + 0.4*clamp01(recency/120)))

    # Engagement final
    engagement = clamp01(base_engagement * (new_mult/cur_mult) * cadence_lift * fatigue_penalty + incentive_boost + personalization_boost)

    # Conversion probability proxy (0..1)
    conversion_base = clamp01(0.06 + 0.22*spend_norm + 0.18*freq_norm + 0.06*prem_norm)
    conversion = clamp01(conversion_base + 0.25*incentive_level*(0.6*disc_norm + 0.2) + 0.18*ai_personalization)

    # Fatigue risk: rises sharply after 3 touches/week and with SMS
    fatigue = clamp01(0.10 + 0.18*max(0.0, touches_per_week-2.0) + (0.10 if new_channel=="sms" else 0.06 if new_channel=="app_push" else 0.04))
    # More engaged segments tolerate slightly more
    fatigue = clamp01(fatigue * (1.05 - 0.15*freq_norm))

    # Unsub risk: tied to fatigue + channel + low personalization
    unsub = clamp01(0.02 + 0.35*fatigue + (0.03 if new_channel=="sms" else 0.01) + 0.05*(1-ai_personalization))

    notes = []
    if touches_per_week > 3.0:
        notes.append("High cadence increases fatigue risk — consider frequency caps.")
    if new_channel == "sms" and ai_personalization < 0.5:
        notes.append("SMS without strong personalization can increase opt-outs.")
    if incentive_level > 0.7 and prem_norm > 0.5:
        notes.append("Heavy incentives may erode premium perception — consider value-add offers instead.")
    if ai_personalization > 0.7:
        notes.append("High personalization can offset fatigue and improve conversion efficiency.")
    if not notes:
        notes.append("Scenario looks balanced. Monitor opt-outs and repeat purchase rate.")

    return {
        "engagement_index": float(engagement),
        "conversion_prob": float(conversion),
        "fatigue_risk": float(fatigue),
        "unsub_risk": float(unsub),
        "notes": notes
    }
