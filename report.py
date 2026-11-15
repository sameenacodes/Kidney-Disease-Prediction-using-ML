"""PDF assessment report (ReportLab)."""
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from ml.common import param_rows

TEAL = colors.HexColor("#0f766e"); INK = colors.HexColor("#0b1b2b"); GREY = colors.HexColor("#64748b")
RISK = {"High": colors.HexColor("#dc2626"), "Moderate": colors.HexColor("#d97706"), "Low": colors.HexColor("#16a34a")}


def build_pdf(a):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=16 * mm,
                            title=f"CKDPredict Assessment #{a['id']}", author="CKDPredict")
    ss = getSampleStyleSheet()
    H = ParagraphStyle("H", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, textColor=INK, alignment=0, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=ss["Normal"], fontSize=9, textColor=GREY)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12, textColor=TEAL, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=ss["Normal"], fontSize=8, leading=11, textColor=GREY)
    r = a["result"]; el = []
    el += [Paragraph("CKDPredict - Kidney Risk Assessment Report", H),
           Paragraph("Clinical decision-support prototype - Machine Learning based risk assessment", sub), Spacer(1, 6)]
    info = [["Assessment ID", f"#{a['id']}", "Date", a["created_at"]],
            ["Patient", a["patient_name"], "Age / Sex", f"{a['age']:g} / {a['sex'] or '-'}" if a["age"] else f"- / {a['sex'] or '-'}"],
            ["Input source", {"manual": "Manual entry", "ocr": "Lab report (OCR, user verified)", "api": "REST API"}.get(a["source"], a["source"]),
             "Model", r.get("model_name", "-")]]
    t = Table(info, colWidths=[28 * mm, 55 * mm, 25 * mm, 62 * mm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9), ("TEXTCOLOR", (0, 0), (0, -1), GREY), ("TEXTCOLOR", (2, 0), (2, -1), GREY),
                           ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"), ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    el += [t, Paragraph("Result", h2)]
    _p = r['probability'] * 100
    pct = ">99.9%" if _p > 99.9 else ("<0.1%" if _p < 0.1 else f"{_p:.1f}%")
    box = Table([[Paragraph(f"<b>{r['prediction']}</b>", ParagraphStyle("p", parent=body, fontSize=14, textColor=INK)),
                  Paragraph(f"Risk level<br/><font size=14 color='{RISK[r['risk']].hexval()}'><b>{r['risk']}</b></font>", body),
                  Paragraph(f"Model probability of CKD<br/><font size=14><b>{pct}</b></font>", body),
                  Paragraph(f"Inputs used<br/><font size=14><b>{int(r['completeness'] * 100)}%</b></font>", body)]],
                colWidths=[62 * mm, 30 * mm, 48 * mm, 30 * mm])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, TEAL), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdfa")),
                             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    el.append(box)
    if r.get("egfr") is not None:
        el += [Spacer(1, 4), Paragraph(f"<b>eGFR (CKD-EPI 2021):</b> {r['egfr']} mL/min/1.73 m2 &nbsp; - &nbsp; <b>KDIGO stage {r['stage']}</b> ({r['stage_text']}). "
                                       "eGFR staging is calculated separately from the ML prediction.", body)]
    for f in r.get("flags", []):
        el.append(Paragraph(f"- {f}", ParagraphStyle("f", parent=body, textColor=colors.HexColor("#b45309"))))
    el.append(Paragraph("Parameters provided", h2))
    rows = [["Parameter", "Value", "Unit", "Reference", ""]]
    for p in param_rows(a["values"]):
        rows.append([p["label"], p["value"], p["unit"], p["range"], {"high": "HIGH", "low": "LOW"}.get(p["flag"], "")])
    pt = Table(rows, colWidths=[62 * mm, 25 * mm, 28 * mm, 35 * mm, 20 * mm], repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), TEAL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
          ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
          ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 3)]
    for i, row in enumerate(rows[1:], 1):
        if row[4]:
            st += [("TEXTCOLOR", (4, i), (4, i), colors.HexColor("#dc2626")), ("FONTNAME", (4, i), (4, i), "Helvetica-Bold")]
    pt.setStyle(TableStyle(st)); el.append(pt)
    if r.get("explanation"):
        el.append(Paragraph("Main factors behind this prediction", h2))
        ex = [["Factor", "Value", "Pushes the estimate", "Share"]] + [[e["label"], f"{e['value']:g}", e["direction"], f"{e['share']}%"] for e in r["explanation"]]
        et = Table(ex, colWidths=[62 * mm, 25 * mm, 55 * mm, 28 * mm])
        et.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0"))]))
        el += [et, Paragraph("Share = relative size of each value's exact contribution (log-odds) to the model score. Tests that were not provided are treated as neutral.", small)]
    el += [Spacer(1, 10), Paragraph(
        "<b>Disclaimer.</b> This report is produced by a machine-learning based clinical decision-support prototype built for academic purposes. "
        "It is a screening / decision-support aid, not a diagnosis, and does not replace a doctor or laboratory testing. The probability is the model's "
        "estimate on the data it was trained on and is not a clinical probability of disease. Results must be confirmed by a qualified medical professional.", small)]
    doc.build(el)
    buf.seek(0)
    return buf
