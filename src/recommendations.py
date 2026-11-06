"""
recommendations.py
------------------
Generates human-readable explanations and contextual guidance for crop
and fertilizer predictions from the trained models.

Important
---------
* No fertilizer dosage quantities are invented here.
* All nutrient ranges cited are general educational references, not
  precise agronomic recommendations.
* Confidence scores are model estimates, not guarantees.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Crop information (general, educational only)
# ---------------------------------------------------------------------------

CROP_INFO: dict[str, dict] = {
    "rice": {
        "season": "Kharif (June – November)",
        "water": "High water requirement",
        "soil": "Clayey or loamy soils with good water retention",
        "notes": "Requires flooded or waterlogged conditions during early growth.",
    },
    "maize": {
        "season": "Kharif / Rabi",
        "water": "Moderate water requirement",
        "soil": "Well-drained loamy soils",
        "notes": "Sensitive to water-logging. Ensure good drainage.",
    },
    "chickpea": {
        "season": "Rabi (October – March)",
        "water": "Low water requirement",
        "soil": "Well-drained sandy loam to loam soils",
        "notes": "Nitrogen-fixing legume; reduces fertilizer need.",
    },
    "kidneybeans": {
        "season": "Kharif",
        "water": "Moderate",
        "soil": "Loamy, well-drained soils",
        "notes": "Legume crop — can fix atmospheric nitrogen.",
    },
    "pigeonpeas": {
        "season": "Kharif",
        "water": "Low to moderate",
        "soil": "Well-drained sandy loam",
        "notes": "Drought-tolerant legume; improves soil fertility.",
    },
    "mothbeans": {
        "season": "Kharif",
        "water": "Very low",
        "soil": "Sandy to loamy",
        "notes": "Highly drought-tolerant; suitable for arid regions.",
    },
    "mungbean": {
        "season": "Kharif / Summer",
        "water": "Low to moderate",
        "soil": "Well-drained loamy soils",
        "notes": "Short-duration legume; good for crop rotation.",
    },
    "blackgram": {
        "season": "Kharif",
        "water": "Moderate",
        "soil": "Well-drained loamy soils",
        "notes": "Legume; adds nitrogen to soil after harvest.",
    },
    "lentil": {
        "season": "Rabi",
        "water": "Low",
        "soil": "Sandy loam to clay loam",
        "notes": "Nitrogen-fixing crop suitable for cool dry conditions.",
    },
    "pomegranate": {
        "season": "Year-round (subtropical)",
        "water": "Low to moderate",
        "soil": "Deep, well-drained loamy soils",
        "notes": "Drought-tolerant fruit crop; suitable for semi-arid regions.",
    },
    "banana": {
        "season": "Year-round (tropical)",
        "water": "High",
        "soil": "Rich loamy soils with good drainage",
        "notes": "High potassium demand; sensitive to water-logging.",
    },
    "mango": {
        "season": "Summer fruiting",
        "water": "Moderate",
        "soil": "Deep, well-drained alluvial soils",
        "notes": "Established trees are drought-tolerant.",
    },
    "grapes": {
        "season": "Winter / Spring",
        "water": "Moderate, well-timed irrigation",
        "soil": "Well-drained sandy to loamy soils",
        "notes": "Requires dry conditions at ripening.",
    },
    "watermelon": {
        "season": "Summer (Zaid)",
        "water": "High during vegetative stage",
        "soil": "Sandy loam",
        "notes": "Requires warm temperatures; intolerant of frost.",
    },
    "muskmelon": {
        "season": "Summer (Zaid)",
        "water": "Moderate",
        "soil": "Sandy loam, well-drained",
        "notes": "Requires hot, dry weather at ripening.",
    },
    "apple": {
        "season": "Temperate regions",
        "water": "Moderate, well-distributed",
        "soil": "Well-drained loamy soils, slightly acidic pH",
        "notes": "Requires chilling hours in winter for proper fruiting.",
    },
    "orange": {
        "season": "Subtropical",
        "water": "Moderate",
        "soil": "Well-drained deep soils",
        "notes": "Sensitive to frost. Requires adequate zinc and iron.",
    },
    "papaya": {
        "season": "Year-round (tropical)",
        "water": "Moderate",
        "soil": "Well-drained sandy loam",
        "notes": "Cannot tolerate water-logging; needs warm climate.",
    },
    "coconut": {
        "season": "Year-round (coastal tropical)",
        "water": "High",
        "soil": "Sandy loam to sandy soils near coast",
        "notes": "Tolerates saline conditions; high potassium demand.",
    },
    "cotton": {
        "season": "Kharif",
        "water": "Moderate to high",
        "soil": "Deep black cotton soils or alluvial soils",
        "notes": "Long growing season; requires warm temperatures.",
    },
    "jute": {
        "season": "Kharif",
        "water": "High; thrives in humid conditions",
        "soil": "Alluvial loamy soils",
        "notes": "Requires warm humid climate; heavy rainfall.",
    },
    "coffee": {
        "season": "Year-round (tropical highlands)",
        "water": "Moderate, well-distributed",
        "soil": "Deep, well-drained laterite or loamy soils",
        "notes": "Shade-grown preferred; sensitive to frost.",
    },
}

# General reference nutrient ranges (educational, based on general agronomy literature)
# Source: General agronomic guidelines. Not crop-specific; always verify locally.
NUTRIENT_RANGES = {
    "N": {"low": 0, "adequate_min": 60, "adequate_max": 140, "high": 200,
          "unit": "kg/ha"},
    "P": {"low": 0, "adequate_min": 20, "adequate_max": 60, "high": 100,
          "unit": "kg/ha"},
    "K": {"low": 0, "adequate_min": 40, "adequate_max": 100, "high": 200,
          "unit": "kg/ha"},
    "ph": {"low": 4.5, "adequate_min": 6.0, "adequate_max": 7.5, "high": 9.0,
           "unit": ""},
    "temperature": {"low": 5, "adequate_min": 15, "adequate_max": 35, "high": 45,
                    "unit": "°C"},
    "humidity": {"low": 10, "adequate_min": 30, "adequate_max": 90, "high": 100,
                 "unit": "%"},
    "rainfall": {"low": 20, "adequate_min": 200, "adequate_max": 2000, "high": 3000,
                 "unit": "mm"},
}


def get_crop_info(crop_name: str) -> dict:
    """Return general information about a crop, or an empty dict."""
    return CROP_INFO.get(crop_name.lower().strip(), {})


def describe_nutrient(col: str, value: float) -> str:
    """Return a simple narrative description for a nutrient value."""
    if col not in NUTRIENT_RANGES:
        return f"{col}: {value}"
    r = NUTRIENT_RANGES[col]
    unit = r["unit"]
    val_str = f"{value:.1f}{' ' + unit if unit else ''}"
    if value < r["adequate_min"]:
        level = "below the general adequate range"
    elif value > r["adequate_max"]:
        level = "above the general adequate range"
    else:
        level = "within the general adequate range"
    return f"{col} = {val_str} ({level})"


def build_crop_explanation(
    input_values: dict,
    crop: str,
    confidence: Optional[float],
    top3: list[dict],
    importance_df=None,
) -> dict:
    """
    Compose a structured explanation for a crop prediction.

    Returns
    -------
    dict with: summary, nutrient_descriptions, crop_info, confidence_note,
               top3_note, feature_importance_note, disclaimer.
    """
    nutrient_descriptions = [
        describe_nutrient(col, val) for col, val in input_values.items()
    ]

    crop_info = get_crop_info(crop)

    if confidence is not None:
        confidence_note = (
            f"The model assigns a probability of {confidence:.1%} to this prediction. "
            "This reflects the model's classification certainty, not a guarantee "
            "of successful cultivation."
        )
    else:
        confidence_note = (
            "Model confidence is not available for this algorithm."
        )

    if top3:
        top3_note = "Top 3 crop suggestions from the model:\n" + "\n".join(
            f"  {i+1}. {item['crop']} — model probability: {item['probability']:.1%}"
            for i, item in enumerate(top3)
        )
    else:
        top3_note = ""

    if importance_df is not None:
        fi_note = (
            "Key factors influencing this prediction (based on feature importance):\n"
            + "\n".join(
                f"  • {row['Feature']}: {row['Importance']:.3f}"
                for _, row in importance_df.head(5).iterrows()
            )
            + "\n\nNote: Feature importance shows statistical patterns in the "
            "training data, not causal relationships."
        )
    else:
        fi_note = ""

    return {
        "summary": f"Based on the provided soil and environmental conditions, "
                   f"the model recommends: **{crop}**.",
        "nutrient_descriptions": nutrient_descriptions,
        "crop_info": crop_info,
        "confidence_note": confidence_note,
        "top3_note": top3_note,
        "feature_importance_note": fi_note,
        "disclaimer": (
            "⚠️ This is a machine learning classification result for educational "
            "purposes. Always verify crop suitability with local agricultural "
            "extension services and professional soil testing before making "
            "planting decisions."
        ),
    }


def build_fertilizer_explanation(
    input_values: dict,
    fertilizer: str,
    confidence: Optional[float],
) -> dict:
    """Compose a structured explanation for a fertilizer prediction."""
    if confidence is not None:
        confidence_note = (
            f"Model confidence: {confidence:.1%}. This is the model's classification "
            "probability, not a measure of agronomic suitability."
        )
    else:
        confidence_note = "Model confidence is not available."

    sustainability_note = (
        "Before applying any fertilizer:\n"
        "• Conduct a laboratory soil test to confirm actual nutrient levels.\n"
        "• Follow the recommended application rates from your local agricultural "
        "  extension office.\n"
        "• Avoid over-application — excess nutrients can leach into water bodies "
        "  and harm the environment.\n"
        "• Consider organic alternatives such as compost or green manures to "
        "  reduce chemical fertilizer dependency.\n"
        "• Split applications are often more effective than single heavy doses."
    )

    return {
        "summary": f"Based on the inputs provided, the model predicts: **{fertilizer}**.",
        "confidence_note": confidence_note,
        "sustainability_note": sustainability_note,
        "disclaimer": (
            "⚠️ This prediction is from a classification model trained on a public "
            "dataset. It does not constitute a professional fertilizer prescription. "
            "No dosage quantities are guaranteed. Consult a certified agronomist "
            "or soil testing laboratory for validated recommendations."
        ),
    }
