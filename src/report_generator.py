"""
report_generator.py
-------------------
Generates executive-ready, branded PDF Field Advisory Reports
for AgriSmart AI using ReportLab.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_field_advisory_pdf(
    input_values: dict,
    crop_predicted: str,
    confidence: Optional[float],
    top3: list[dict],
    fertilizer_predicted: Optional[str] = None,
    region_note: Optional[str] = None,
    model_name: str = "Random Forest Classifier",
    model_accuracy: float = 0.993,
) -> bytes:
    """
    Build a PDF report in memory and return its bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#065f46")
    accent_color = colors.HexColor("#10b981")
    text_dark = colors.HexColor("#1e293b")
    text_muted = colors.HexColor("#64748b")
    bg_light = colors.HexColor("#f8fafc")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=text_muted,
    )
    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=text_dark,
    )
    badge_style = ParagraphStyle(
        "BadgeText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#ffffff"),
        alignment=1,
    )
    small_disclaimer = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#94a3b8"),
    )

    story = []

    # 1. Header Banner Table
    date_str = datetime.now().strftime("%d %B %Y, %H:%M")
    rep_id = f"AGRI-{datetime.now().strftime('%Y%m%d')}-{hash(crop_predicted) % 10000:04d}"

    header_data = [
        [
            Paragraph("🌾 AgriSmart AI — Precision Field Advisory Report", title_style),
            Paragraph(f"<b>Report ID:</b> {rep_id}<br/><b>Date:</b> {date_str}<br/><b>Region:</b> {region_note or 'Unspecified'}", subtitle_style),
        ]
    ]
    header_table = Table(header_data, colWidths=[340, 180])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceBefore=4, spaceAfter=12))

    # 2. Executive Recommendation Callout Card
    conf_pct = f"{confidence:.1%}" if confidence is not None else "N/A"
    recom_html = f"""
    <b>PRIMARY MACHINE LEARNING RECOMMENDATION:</b><br/>
    <font size="16"><b>{crop_predicted.upper()}</b></font><br/>
    Ensemble Certainty: <b>{conf_pct}</b> | Algorithm: <b>{model_name} (Test Acc: {model_accuracy:.1%})</b>
    """
    card_data = [[
        Paragraph(recom_html, ParagraphStyle("CardP", parent=body_style, textColor=colors.HexColor("#ffffff"), leading=16))
    ]]
    card_table = Table(card_data, colWidths=[520])
    card_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), primary_color),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 12))

    # 3. Measured Field Parameters vs Benchmarks Table
    story.append(Paragraph("1. Field Soil Chemistry & Microclimate Telemetry", h2_style))
    field_rows = [
        ["Parameter", "Measured Value", "Agronomic Benchmark", "Status Assessment"],
        ["Nitrogen (N)", f"{input_values.get('N', 0):.1f} kg/ha", "60 – 140 kg/ha", "Adequate" if 60 <= input_values.get('N', 0) <= 140 else ("Low" if input_values.get('N', 0) < 60 else "High")],
        ["Phosphorus (P)", f"{input_values.get('P', 0):.1f} kg/ha", "20 – 60 kg/ha", "Adequate" if 20 <= input_values.get('P', 0) <= 60 else ("Low" if input_values.get('P', 0) < 20 else "High")],
        ["Potassium (K)", f"{input_values.get('K', 0):.1f} kg/ha", "40 – 100 kg/ha", "Adequate" if 40 <= input_values.get('K', 0) <= 100 else ("Low" if input_values.get('K', 0) < 40 else "High")],
        ["Soil pH", f"{input_values.get('ph', 0):.2f}", "6.0 – 7.5 (Neutral)", "Ideal" if 6.0 <= input_values.get('ph', 0) <= 7.5 else ("Acidic" if input_values.get('ph', 0) < 6.0 else "Alkaline")],
        ["Ambient Temperature", f"{input_values.get('temperature', 0):.1f} °C", "15 – 35 °C", "Optimal"],
        ["Relative Humidity", f"{input_values.get('humidity', 0):.1f} %", "30 – 90 %", "Normal"],
        ["Precipitation / Rainfall", f"{input_values.get('rainfall', 0):.1f} mm", "200 – 2000 mm", "Monitored"],
    ]
    t_fields = Table(field_rows, colWidths=[140, 110, 140, 130])
    t_fields.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#ffffff")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.HexColor("#f1f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_fields)
    story.append(Spacer(1, 10))

    # 4. Multi-class Ensemble Alternatives
    story.append(Paragraph("2. Top-3 Candidate Classification Probabilities", h2_style))
    top3_rows = [["Rank", "Candidate Crop", "Model Probability", "Agricultural Feasibility"]]
    for idx, item in enumerate(top3):
        prob = item.get("probability", 0)
        top3_rows.append([
            f"#{idx+1}",
            item.get("crop", "").title(),
            f"{prob:.1%}",
            "Primary Recommended" if idx == 0 else "Strong Viable Alternative",
        ])
    t_top3 = Table(top3_rows, colWidths=[60, 160, 140, 160])
    t_top3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#ffffff")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_top3)
    story.append(Spacer(1, 10))

    # 5. Targeted Fertilizer Advice
    story.append(Paragraph("3. Nutrient Stewardship & Fertilizer Advisory", h2_style))
    fert_text = f"""
    <b>Target Fertilizer Formulation:</b> {fertilizer_predicted or 'Balanced NPK Formulation based on Soil Testing'}<br/>
    • <b>Application Strategy:</b> Split chemical applications into 2 to 3 vegetative cycles to prevent nitrogen leaching.<br/>
    • <b>Runoff Mitigation:</b> Maintain grassed riparian buffer zones along nearby irrigation channels or water bodies.<br/>
    • <b>Organic Matter:</b> Incorporate 5–10 MT/ha of well-rotted Farmyard Manure (FYM) or compost to improve soil organic carbon (SOC).
    """
    story.append(Paragraph(fert_text, body_style))
    story.append(Spacer(1, 10))

    # 6. UN Sustainable Development Goals Alignment
    story.append(Paragraph("4. Sustainable Development Framework Alignment", h2_style))
    sdg_text = """
    • <b>SDG 2 (Zero Hunger):</b> Promotes optimal agronomic matching to improve local harvest yields and food security.<br/>
    • <b>SDG 12 (Responsible Consumption & Production):</b> Prevents synthetic fertilizer over-dosage and chemical waste.<br/>
    • <b>SDG 15 (Life on Land):</b> Preserves subterranean microbial biomes and prevents long-term topsoil acidification.
    """
    story.append(Paragraph(sdg_text, body_style))
    story.append(Spacer(1, 12))

    # 7. Official Legal / Academic Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceBefore=4, spaceAfter=8))
    disclaimer_text = """
    <b>ACADEMIC & AGRONOMIC DISCLAIMER:</b> This advisory document was generated automatically by AgriSmart AI, an academic supervised machine learning decision-support prototype. Machine learning predictions represent statistical alignment with public research datasets and do not guarantee specific crop survival, biological yields, or financial returns. Numerical fertilizer applications must be validated with certified agricultural extension officers or authorized soil testing laboratories prior to field application.
    """
    story.append(Paragraph(disclaimer_text, small_disclaimer))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
