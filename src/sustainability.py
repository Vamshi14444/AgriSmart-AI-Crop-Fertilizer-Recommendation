"""
sustainability.py
-----------------
Educational sustainability guidance and checklist for AgriSmart AI.

All content here is general agricultural education. It is clearly
distinguished from ML model predictions throughout the application.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Sustainability tips
# ---------------------------------------------------------------------------

SUSTAINABILITY_TIPS = [
    {
        "category": "🌱 Soil Health",
        "tip": "Conduct regular soil tests (every 2–3 years) to understand actual "
               "nutrient levels before purchasing fertilizers. This avoids unnecessary "
               "cost and environmental harm.",
    },
    {
        "category": "⚖️ Balanced Nutrients",
        "tip": "Apply nutrients based on soil test results and crop requirements. "
               "Over-application of nitrogen can lead to nitrate leaching, which "
               "contaminates groundwater.",
    },
    {
        "category": "🔄 Crop Rotation",
        "tip": "Rotate crops seasonally to break pest cycles, improve soil structure, "
               "and reduce dependence on synthetic fertilizers. Legumes fix atmospheric "
               "nitrogen and benefit the next crop.",
    },
    {
        "category": "♻️ Organic Matter",
        "tip": "Incorporate compost, farmyard manure, or crop residues into the soil. "
               "Organic matter improves soil structure, water retention, and biological "
               "activity.",
    },
    {
        "category": "💧 Water Conservation",
        "tip": "Use drip irrigation or sprinkler systems to reduce water waste. "
               "Irrigate based on crop stage and local evapotranspiration data, "
               "not fixed schedules.",
    },
    {
        "category": "🌿 Reduce Runoff",
        "tip": "Avoid fertilizer application before heavy rain. Maintain vegetated "
               "buffer strips around water bodies to reduce nutrient runoff and "
               "protect aquatic ecosystems.",
    },
    {
        "category": "🐛 Integrated Pest Management",
        "tip": "Combine biological controls, resistant varieties, and targeted pesticide "
               "use to manage pests. Reducing pesticide overuse protects soil organisms "
               "and pollinators.",
    },
    {
        "category": "🌾 Crop Diversification",
        "tip": "Growing a variety of crops reduces economic risk, supports local "
               "biodiversity, and creates more resilient farming systems.",
    },
    {
        "category": "📏 Split Fertilizer Applications",
        "tip": "Divide fertilizer applications into two or three doses instead of "
               "applying all at once. This matches nutrient supply to crop demand "
               "and reduces losses.",
    },
    {
        "category": "🌍 SDG Alignment",
        "tip": "Sustainable farming practices directly support SDG 2 (Zero Hunger), "
               "SDG 12 (Responsible Consumption and Production), and SDG 15 (Life "
               "on Land). Small farm-level decisions contribute to global goals.",
    },
]


# ---------------------------------------------------------------------------
# Sustainability checklist
# ---------------------------------------------------------------------------

SUSTAINABILITY_CHECKLIST = [
    ("Soil Testing", "Have I tested my soil for NPK and pH in the past 2–3 years?"),
    ("Balanced Nutrients", "Am I applying fertilizer based on test results, not guesses?"),
    ("Water Efficiency", "Am I using efficient irrigation and avoiding waterlogging?"),
    ("Crop Rotation", "Have I planned crop rotation to restore soil health?"),
    ("Organic Matter", "Am I adding compost or organic matter to my soil?"),
    ("Runoff Prevention", "Am I avoiding fertilizer application before heavy rain?"),
    ("Certified Advice", "Have I consulted an agricultural extension officer this season?"),
    ("Buffer Zones", "Are vegetated buffer strips maintained near water bodies?"),
    ("Seed Selection", "Am I using locally adapted or certified seed varieties?"),
    ("Record Keeping", "Am I keeping records of inputs, weather, and crop performance?"),
]


# ---------------------------------------------------------------------------
# SDG descriptions
# ---------------------------------------------------------------------------

SDG_RELEVANCE = {
    "SDG 2 — Zero Hunger": (
        "By helping farmers select crops suited to their soil and climate conditions, "
        "this tool supports more productive and reliable food production. Better-informed "
        "crop choices can reduce crop failure and improve food security at the local level."
    ),
    "SDG 12 — Responsible Consumption and Production": (
        "The fertilizer recommendation module encourages farmers to match inputs to "
        "actual crop needs, discouraging over-application of synthetic fertilizers. "
        "This reduces waste, lowers input costs, and decreases environmental pollution."
    ),
    "SDG 15 — Life on Land": (
        "Sustainable soil management, crop rotation, and reduced chemical runoff "
        "directly protect terrestrial ecosystems. This tool promotes practices that "
        "preserve soil biodiversity and prevent land degradation."
    ),
}


def get_all_tips() -> list[dict]:
    return SUSTAINABILITY_TIPS


def get_checklist() -> list[tuple[str, str]]:
    return SUSTAINABILITY_CHECKLIST


def get_sdg_relevance() -> dict[str, str]:
    return SDG_RELEVANCE
