"""
Module 7: Rule-Based, Interpretable Personalised Health Advisory Engine
Translates forecasted AQI + 80% uncertainty interval into transparent, rule-triggered guidance
for 5 distinct population profiles:
  1. General Public
  2. Asthmatics & Respiratory Patients
  3. Children & School-Age Youth
  4. Elderly (65+ years)
  5. Outdoor Workers & Athletes
"""

from dataclasses import dataclass, asdict
from typing import List, Dict


CPCB_BANDS = [
    (0, 50, "Good", "#2E7D32"),
    (51, 100, "Satisfactory", "#689F38"),
    (101, 200, "Moderate", "#FBC02D"),
    (201, 300, "Poor", "#F57C00"),
    (301, 400, "Very Poor", "#D32F2F"),
    (401, 500, "Severe", "#7B1FA2"),
]


def classify_aqi_band(aqi_value: float) -> Dict[str, str]:
    val = max(0.0, min(500.0, float(aqi_value)))
    for low, high, label, color in CPCB_BANDS:
        if low <= val <= high:
            return {"band": label, "color": color, "range": f"{low}-{high}"}
    return {"band": "Severe", "color": "#7B1FA2", "range": "401-500"}


@dataclass
class HealthAdvisoryOutput:
    group: str
    forecast_median: float
    lower_80: float
    upper_80: float
    predicted_band: str
    worst_case_band: str
    risk_level: str
    uncertainty_note: str
    triggered_rules: List[str]
    recommended_actions: List[str]
    protective_gear: str
    outdoor_activity_limit: str

    def to_dict(self) -> dict:
        return asdict(self)


def generate_health_advisory(
    group: str,
    forecast_median: float,
    lower_80: float,
    upper_80: float,
    dominant_pollutant: str = "PM2.5",
    wind_speed: float = 2.0,
) -> HealthAdvisoryOutput:
    """
    Deterministic, auditable rule engine incorporating both central forecast (median)
    and upper uncertainty bound (90th percentile) to avoid false reassurance.
    """
    med_info = classify_aqi_band(forecast_median)
    upper_info = classify_aqi_band(upper_80)
    interval_width = upper_80 - lower_80

    triggered_rules = [
        f"RULE_BAND_MEDIAN: Central AQI {forecast_median:.0f} falls in '{med_info['band']}' ({med_info['range']})."
    ]

    # Uncertainty communication logic
    if upper_info["band"] != med_info["band"]:
        uncertainty_note = (
            f"Moderate-to-High Forecast Uncertainty (±{interval_width/2:.0f} AQI): "
            f"While central estimate is '{med_info['band']}' ({forecast_median:.0f}), "
            f"upper 90th-percentile bound reaches {upper_80:.0f} ('{upper_info['band']}'). "
            f"Precautions are escalated by one tier for vulnerable groups."
        )
        triggered_rules.append(
            f"RULE_UNCERTAINTY_ESCALATION: Upper bound ({upper_80:.0f}) crosses into '{upper_info['band']}'."
        )
        effective_aqi = 0.65 * forecast_median + 0.35 * upper_80
    else:
        uncertainty_note = (
            f"High Forecast Confidence (80% PI: {lower_80:.0f}–{upper_80:.0f}): "
            f"Both central and upper bounds remain within '{med_info['band']}'."
        )
        triggered_rules.append("RULE_CONFIDENCE_STABLE: 80% interval stays inside single CPCB category.")
        effective_aqi = forecast_median

    if wind_speed < 1.5:
        triggered_rules.append(
            f"RULE_STAGNANT_AIR: Low wind speed ({wind_speed:.1f} m/s) traps {dominant_pollutant} near ground level."
        )

    # Group-specific sensitivity thresholds
    group_key = group.strip().lower()

    if "asthma" in group_key or "respiratory" in group_key:
        triggered_rules.append("RULE_GROUP_ASTHMATIC: Strict threshold applied (advisory triggers at Effective AQI > 90).")
        if effective_aqi <= 90:
            risk = "Low-Moderate"
            actions = [
                "Normal outdoor routine allowed; carry rescue inhaler (SABA) as standard precaution.",
                "Monitor morning symptoms if humidity exceeds 75%."
            ]
            gear = "Optional surgical/KN95 mask in heavy traffic"
            limit = "No strict duration cap"
        elif effective_aqi <= 180:
            risk = "High"
            actions = [
                "Avoid vigorous outdoor cardio near main roads during rush hours (8–10 AM, 6–9 PM).",
                "Pre-medicate only if prescribed by physician before unavoidable outdoor transit.",
                "Keep bedroom windows closed at night."
            ]
            gear = "Certified N95/FFP2 respirator outdoors"
            limit = "Max 45–60 mins light outdoor exposure"
        else:
            risk = "Critical"
            actions = [
                "Remain indoors with HEPA air purifier running on medium/high.",
                "Follow written Asthma Action Plan; check peak expiratory flow twice daily.",
                "Seek medical care immediately if wheezing, chest tightness, or nocturnal cough occurs."
            ]
            gear = "Fit-tested N95 respirator mandatory if stepping outside"
            limit = "Avoid all non-essential outdoor exposure (< 15 mins)"

    elif "child" in group_key:
        triggered_rules.append("RULE_GROUP_CHILDREN: Higher minute ventilation per body weight (triggers at Effective AQI > 100).")
        if effective_aqi <= 100:
            risk = "Low"
            actions = ["Outdoor school sports and playground activities are safe."]
            gear = "None required"
            limit = "Unrestricted"
        elif effective_aqi <= 200:
            risk = "Moderate-High"
            actions = [
                "Move physical education (PE) classes and recess indoors or shorten intensity.",
                "Ensure adequate hydration and nasal saline rinse after school commute."
            ]
            gear = "Child-sized KN95/N95 mask during bus/auto commute"
            limit = "Max 60 mins moderate outdoor play"
        else:
            risk = "Severe"
            actions = [
                "Cancel all outdoor school assemblies, sports, and playground time.",
                "Keep indoor classrooms sealed with air filtration active."
            ]
            gear = "Well-fitted pediatric N95 mask for transit only"
            limit = "Zero outdoor sports; transit only"

    elif "elderly" in group_key or "senior" in group_key:
        triggered_rules.append("RULE_GROUP_ELDERLY: Elevated cardiopulmonary risk profile (triggers at Effective AQI > 100).")
        if effective_aqi <= 100:
            risk = "Low"
            actions = ["Morning or evening walks are safe; stay hydrated."]
            gear = "None required"
            limit = "Unrestricted"
        elif effective_aqi <= 200:
            risk = "High"
            actions = [
                "Shift morning walks to late afternoon (3–5 PM) when boundary layer height peaks.",
                "Monitor blood pressure and heart rate; avoid strenuous gardening or stair climbing outdoors."
            ]
            gear = "N95 mask in roadside areas"
            limit = "Max 30–45 mins gentle walk"
        else:
            risk = "Severe"
            actions = [
                "Stay indoors; perform light indoor mobility exercises instead of outdoor walks.",
                "Avoid burning incense, candles, or indoor smoke sources.",
                "Contact physician if shortness of breath, dizziness, or palpitations occur."
            ]
            gear = "N95 respirator if exiting building"
            limit = "Stay indoors"

    elif "outdoor" in group_key or "worker" in group_key or "athlete" in group_key:
        triggered_rules.append("RULE_GROUP_OUTDOOR_WORKER: High cumulative inhalation dose due to prolonged exposure.")
        if effective_aqi <= 100:
            risk = "Low"
            actions = ["Standard shift schedule; maintain hydration."]
            gear = "Optional dust mask at construction/road sites"
            limit = "Normal 8-hour shift"
        elif effective_aqi <= 200:
            risk = "Moderate-High"
            actions = [
                "Schedule heavy physical labor between 12 PM and 4 PM (lowest pollution trapping).",
                "Take 15-minute indoor/filtered rest breaks every 2 hours."
            ]
            gear = "N95/FFP2 respirator with exhalation valve for comfort during exertion"
            limit = "Rotate high-exertion tasks every 2 hours"
        else:
            risk = "Severe"
            actions = [
                "Mandatory N95/FFP2 respirator for all outdoor duty.",
                "Enforce 20-minute clean-air break every 90 minutes; halt non-urgent heavy outdoor exertion."
            ]
            gear = "Mandatory N95/FFP2 respirator (replace when damp/soiled)"
            limit = "Cap heavy outdoor exertion to <= 2 hours total"

    else:
        # General Public
        triggered_rules.append("RULE_GROUP_GENERAL: Standard CPCB public health thresholds applied.")
        if effective_aqi <= 100:
            risk = "Low"
            actions = ["Air quality is acceptable for all normal outdoor activities."]
            gear = "None required"
            limit = "Unrestricted"
        elif effective_aqi <= 200:
            risk = "Moderate"
            actions = [
                "Unusually sensitive individuals should reduce prolonged outdoor exertion.",
                "Keep car windows closed with cabin air on recirculate during traffic."
            ]
            gear = "Optional N95 for prolonged roadside exposure"
            limit = "Normal activity; reduce extreme endurance workouts"
        elif effective_aqi <= 300:
            risk = "High"
            actions = [
                "Move intense outdoor workouts (running, cycling) indoors.",
                "Use wet mopping indoors instead of dry sweeping to avoid resuspending dust."
            ]
            gear = "N95 mask recommended outdoors"
            limit = "Max 60 mins outdoor exercise"
        else:
            risk = "Severe"
            actions = [
                "Avoid all outdoor physical exertion; keep doors and windows sealed.",
                "Run HEPA air purifiers indoors and avoid indoor smoke/frying fumes."
            ]
            gear = "N95/FFP2 respirator mandatory outdoors"
            limit = "Minimize all outdoor time"

    return HealthAdvisoryOutput(
        group=group,
        forecast_median=round(forecast_median, 1),
        lower_80=round(lower_80, 1),
        upper_80=round(upper_80, 1),
        predicted_band=med_info["band"],
        worst_case_band=upper_info["band"],
        risk_level=risk,
        uncertainty_note=uncertainty_note,
        triggered_rules=triggered_rules,
        recommended_actions=actions,
        protective_gear=gear,
        outdoor_activity_limit=limit,
    )
