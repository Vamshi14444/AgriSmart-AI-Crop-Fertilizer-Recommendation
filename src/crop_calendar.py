"""
crop_calendar.py
----------------
Seasonal Agronomic Crop Calendar (Kharif, Rabi, Zaid, Perennial)
and interactive Plotly visual timeline generator.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Seasonal classification and month windows (1=Jan, 12=Dec)
CROP_CALENDAR: dict[str, dict] = {
    "rice": {
        "season_type": "Kharif (Monsoon)",
        "sowing_months": ["June", "July"],
        "growing_months": ["August", "September"],
        "harvest_months": ["October", "November"],
        "description": "Sown at the onset of the southwest monsoon; requires standing water during tillering.",
    },
    "maize": {
        "season_type": "Kharif / Rabi",
        "sowing_months": ["June", "July"],
        "growing_months": ["August", "September"],
        "harvest_months": ["September", "October"],
        "description": "Primarily Kharif, though grown in Rabi under assured irrigation in southern and central zones.",
    },
    "chickpea": {
        "season_type": "Rabi (Winter)",
        "sowing_months": ["October", "November"],
        "growing_months": ["December", "January"],
        "harvest_months": ["February", "March"],
        "description": "Thrives in post-monsoon residual soil moisture and cool, dry winter weather.",
    },
    "kidneybeans": {
        "season_type": "Kharif / Rabi (Hills)",
        "sowing_months": ["May", "June"],
        "growing_months": ["July", "August"],
        "harvest_months": ["September", "October"],
        "description": "Grown in mid and high hills during summer/kharif; sensitive to excessive frost.",
    },
    "pigeonpeas": {
        "season_type": "Kharif (Long Duration)",
        "sowing_months": ["June", "July"],
        "growing_months": ["August", "September", "October", "November"],
        "harvest_months": ["December", "January"],
        "description": "Long-duration pulse often intercropped with sorghum, maize, or soybean.",
    },
    "mothbeans": {
        "season_type": "Kharif (Arid)",
        "sowing_months": ["July"],
        "growing_months": ["August"],
        "harvest_months": ["September", "October"],
        "description": "Short duration drought-hardy crop cultivated with early monsoon rainfall in desert zones.",
    },
    "mungbean": {
        "season_type": "Zaid / Kharif",
        "sowing_months": ["March", "July"],
        "growing_months": ["April", "August"],
        "harvest_months": ["May", "September"],
        "description": "Fast 60-day crop grown in spring/summer (Zaid) or early Kharif.",
    },
    "blackgram": {
        "season_type": "Kharif / Rabi",
        "sowing_months": ["June", "July"],
        "growing_months": ["August"],
        "harvest_months": ["September", "October"],
        "description": "Warm-season pulse that also succeeds in post-rice residual moisture.",
    },
    "lentil": {
        "season_type": "Rabi (Winter)",
        "sowing_months": ["October", "November"],
        "growing_months": ["December", "January"],
        "harvest_months": ["February", "March"],
        "description": "Cool season legume utilizing winter dew and light post-monsoon rain.",
    },
    "watermelon": {
        "season_type": "Zaid (Summer)",
        "sowing_months": ["January", "February"],
        "growing_months": ["March", "April"],
        "harvest_months": ["May", "June"],
        "description": "Hot dry weather during ripening promotes high sugar accumulation in fruits.",
    },
    "muskmelon": {
        "season_type": "Zaid (Summer)",
        "sowing_months": ["February", "March"],
        "growing_months": ["April"],
        "harvest_months": ["May", "June"],
        "description": "Demands warm temperatures and dry atmospheric conditions during fruit maturation.",
    },
    "cotton": {
        "season_type": "Kharif (Commercial)",
        "sowing_months": ["May", "June"],
        "growing_months": ["July", "August", "September"],
        "harvest_months": ["October", "November", "December"],
        "description": "Long duration fiber crop requiring warm sunny weather during boll opening and picking.",
    },
    "jute": {
        "season_type": "Pre-Kharif / Kharif",
        "sowing_months": ["March", "April"],
        "growing_months": ["May", "June"],
        "harvest_months": ["July", "August"],
        "description": "Requires humid warm conditions and access to slow-moving water pools for fiber retting.",
    },
    "apple": {
        "season_type": "Perennial (Temperate)",
        "sowing_months": ["December", "January"],
        "growing_months": ["March", "April", "May", "June"],
        "harvest_months": ["July", "August", "September"],
        "description": "Winter dormant deciduous fruit; blooms in spring with harvest across late summer.",
    },
    "orange": {
        "season_type": "Perennial (Subtropical)",
        "sowing_months": ["July", "August"],
        "growing_months": ["September", "October"],
        "harvest_months": ["November", "December", "January"],
        "description": "Ambe bahar or Mrig bahar flush; primary fruit maturation happens through cool winter.",
    },
    "banana": {
        "season_type": "Perennial (12-Month Cycle)",
        "sowing_months": ["June", "September"],
        "growing_months": ["Year-round"],
        "harvest_months": ["Year-round"],
        "description": "Tropical giant herb yielding continuous bunches 11–13 months post sucker planting.",
    },
    "mango": {
        "season_type": "Perennial (Summer Fruiting)",
        "sowing_months": ["July", "August"],
        "growing_months": ["October", "November", "December"],
        "harvest_months": ["April", "May", "June"],
        "description": "Flower panicles emerge in January-February, ripening under peak summer sunshine.",
    },
    "grapes": {
        "season_type": "Perennial (Winter/Spring Harvest)",
        "sowing_months": ["October"],
        "growing_months": ["November", "December", "January"],
        "harvest_months": ["February", "March", "April"],
        "description": "Forward pruning in October yields sweet bunches ready for table & wine by spring.",
    },
    "pomegranate": {
        "season_type": "Perennial (Multiple Bahars)",
        "sowing_months": ["June", "July"],
        "growing_months": ["September", "October"],
        "harvest_months": ["December", "January", "February"],
        "description": "Regulated flowering (Mrig / Hasta / Ambe bahar) enables flexible market targeting.",
    },
    "papaya": {
        "season_type": "Perennial (Year-round)",
        "sowing_months": ["June", "July", "September"],
        "growing_months": ["Year-round"],
        "harvest_months": ["Year-round (from Month 9)"],
        "description": "Fast growing fruit bearing continuous flowers and fruit columns along the trunk.",
    },
    "coconut": {
        "season_type": "Perennial (Evergreen)",
        "sowing_months": ["May", "June"],
        "growing_months": ["Year-round"],
        "harvest_months": ["Year-round (Monthly picks)"],
        "description": "Mature palms open a new inflorescence every 28–30 days, giving monthly nut harvests.",
    },
    "coffee": {
        "season_type": "Perennial (Winter Harvest)",
        "sowing_months": ["June", "July"],
        "growing_months": ["August", "September", "October"],
        "harvest_months": ["November", "December", "January"],
        "description": "Blossom showers trigger flowering in March; red ripe berries picked across winter.",
    },
}

MONTHS_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_NAME_MAP = {
    "January": "Jan", "February": "Feb", "March": "Mar", "April": "Apr",
    "May": "May", "June": "Jun", "July": "Jul", "August": "Aug",
    "September": "Sep", "October": "Oct", "November": "Nov", "December": "Dec"
}


def get_calendar_info(crop_id: str) -> dict:
    return CROP_CALENDAR.get(crop_id.lower().strip(), {})


def build_crop_timeline_chart(selected_crops: list[str]) -> go.Figure:
    """
    Build a horizontal seasonal timeline chart for selected crops across 12 months.
    """
    records = []
    for crop in selected_crops:
        info = get_calendar_info(crop)
        if not info:
            continue
        c_title = crop.title()

        # Map months
        sow = [MONTH_NAME_MAP.get(m, m[:3]) for m in info.get("sowing_months", [])]
        grow = [MONTH_NAME_MAP.get(m, m[:3]) for m in info.get("growing_months", [])]
        harv = [MONTH_NAME_MAP.get(m, m[:3]) for m in info.get("harvest_months", [])]

        for m in MONTHS_ORDER:
            stage = "Dormant / Off-Season"
            if m in sow:
                stage = "Sowing / Planting"
            elif m in harv:
                stage = "Harvesting Window"
            elif m in grow or "Year-round" in info.get("growing_months", []):
                stage = "Vegetative Growth"

            records.append({
                "Crop": c_title,
                "Month": m,
                "Stage": stage,
            })

    df = pd.DataFrame(records)
    if df.empty:
        return go.Figure()

    color_map = {
        "Sowing / Planting": "#10b981",    # Emerald
        "Vegetative Growth": "#06b6d4",    # Cyan
        "Harvesting Window": "#f59e0b",    # Amber
        "Dormant / Off-Season": "rgba(255, 255, 255, 0.05)",
    }

    fig = px.scatter(
        df,
        x="Month",
        y="Crop",
        color="Stage",
        color_discrete_map=color_map,
        size=[18] * len(df),
        symbol="Stage",
        symbol_map={
            "Sowing / Planting": "circle",
            "Vegetative Growth": "square",
            "Harvesting Window": "diamond",
            "Dormant / Off-Season": "circle-open",
        },
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Plus Jakarta Sans"),
        height=max(320, len(selected_crops) * 45),
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(
            categoryorder="array",
            categoryarray=MONTHS_ORDER,
            gridcolor="rgba(255,255,255,0.06)",
        ),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig
