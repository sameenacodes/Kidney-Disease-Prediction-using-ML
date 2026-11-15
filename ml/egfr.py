"""eGFR (CKD-EPI 2021, race-free creatinine equation) and KDIGO G-stage."""

def egfr_ckdepi_2021(scr_mg_dl, age, sex):
    """sex: 'M' or 'F'. Returns mL/min/1.73 m2 or None if inputs are missing/invalid."""
    if scr_mg_dl is None or age is None or sex not in ("M", "F") or scr_mg_dl <= 0 or age <= 0:
        return None
    female = sex == "F"
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    ratio = scr_mg_dl / kappa
    val = 142 * (min(ratio, 1) ** alpha) * (max(ratio, 1) ** -1.200) * (0.9938 ** age)
    if female:
        val *= 1.012
    return round(val, 1)


def g_stage(egfr):
    if egfr is None:
        return None
    if egfr >= 90: return "G1"
    if egfr >= 60: return "G2"
    if egfr >= 45: return "G3a"
    if egfr >= 30: return "G3b"
    if egfr >= 15: return "G4"
    return "G5"

STAGE_TEXT = {
    "G1": "Normal or high kidney function (kidney damage markers needed for CKD)",
    "G2": "Mildly decreased",
    "G3a": "Mildly to moderately decreased",
    "G3b": "Moderately to severely decreased",
    "G4": "Severely decreased",
    "G5": "Kidney failure range",
}
