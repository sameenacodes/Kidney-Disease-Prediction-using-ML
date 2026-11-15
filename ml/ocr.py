"""Lab-report reader: PDF (text layer or scanned) / image -> candidate values for the form.
The user always reviews the extracted values before prediction."""
import os, re, shutil
import numpy as np

NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
QUAL = {"nil": 0, "negative": 0, "neg": 0, "absent": 0, "not detected": 0, "trace": 1, "present": 1,
        "+": 1, "1+": 1, "++": 2, "2+": 2, "+++": 3, "3+": 3, "++++": 4, "4+": 4}

# key -> (name regex, exclusion regex or None)
NAMES = {
    "hemo": (r"h(?:a)?emoglobin|\bhb\b|\bhgb\b", r"a1c|glycosylated|glycated"),
    "sc":   (r"(?:serum\s+)?creatinine", r"clearance|urine|ratio|acr|egfr"),
    "bu":   (r"blood\s+urea(?!\s+nitrogen)|\burea\b|\bbun\b|urea\s+nitrogen", r"urine"),
    "sod":  (r"sodium|\bna\b\+?", None),
    "pot":  (r"potassium|\bk\b\+?", None),
    "pcv":  (r"packed\s+cell\s+volume|\bpcv\b|h(?:a)?ematocrit|\bhct\b", None),
    "rc":   (r"(?:t[oe]tal\s+)?rbc\s+count|red\s+blood\s+cell\s+count|r\.b\.c\.?\s+count|erythrocyte\s+count|\brbc\b(?!.*hpf)", r"hpf|urine"),
    "wc":   (r"(?:total\s+)?(?:wbc|w\.b\.c\.?|leuc?ocyte|white\s+blood\s+cell)s?\s*(?:count)?|tlc", r"hpf|urine"),
    "bgr":  (r"random\s+(?:blood\s+)?(?:sugar|glucose)|\brbs\b|blood\s+glucose|blood\s+sugar", r"fasting|post|urine|hba1c"),
    "sg":   (r"specific\s+gravity|\bsp\.?\s*gr\.?\b", None),
    "bp":   (r"blood\s+pressure|\bbp\b", None),
}
RANGE = {"hemo": (3, 22), "sc": (0.1, 25), "bu": (1, 400), "sod": (90, 190), "pot": (1.5, 9), "pcv": (10, 65)}


def _num(s):
    return float(s.replace(",", ""))


def read_texts(path):
    """Returns (list_of_text_variants, method). Scanned inputs yield several OCR variants that are voted on later."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for pg in pdf.pages[:6]:
                text += (pg.extract_text() or "") + "\n"
        if len(text.strip()) >= 40:
            return [text], "pdf-text"
        texts = []
        with pdfplumber.open(path) as pdf:
            imgs = [np.array(pg.to_image(resolution=220).original.convert("RGB")) for pg in pdf.pages[:3]]
        for variant in range(len(VARIANTS)):
            texts.append("\n".join(_ocr_image(i, variant) for i in imgs))
        return texts, "pdf-ocr"
    from PIL import Image
    rgb = np.array(Image.open(path).convert("RGB"))
    return [_ocr_image(rgb, v) for v in range(len(VARIANTS))], "image-ocr"


# (preprocess, target width px, tesseract page-segmentation mode)
VARIANTS = [("adaptive", 2400, 6), ("gray", 3200, 6), ("adaptive", 1600, 4)]


def _ocr_image(rgb, variant=0):
    import cv2, pytesseract
    cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not cmd:
        raise RuntimeError("Tesseract OCR is not installed. Install it or upload a text-based PDF.")
    pytesseract.pytesseract.tesseract_cmd = cmd
    kind, target, psm = VARIANTS[variant]
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    s = target / g.shape[1]
    if abs(s - 1) > 0.05:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    if kind == "adaptive":
        g = cv2.fastNlMeansDenoising(g, None, 12, 7, 21)
        g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15)
    return pytesseract.image_to_string(g, config=f"--oem 3 --psm {psm}")


def _first_number(after):
    m = re.search(NUM, after)
    return _num(m.group(0)) if m else None


def parse_report(text):
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in text.splitlines() if l.strip()]
    found, notes = {}, {}

    # ---- demographics
    blob = "\n".join(lines)
    m = re.search(r"age\s*/\s*sex\s*[:\-]?\s*(\d{1,3})\s*(?:y\w*)?\s*/?\s*(male|female|m|f)\b", blob, re.I)
    if m:
        found["age"] = float(m.group(1)); found["sex"] = m.group(2)[0].upper()
    else:
        m = re.search(r"\bage\s*[:\-]?\s*(\d{1,3})", blob, re.I)
        if m: found["age"] = float(m.group(1))
        m = re.search(r"\bsex\s*[:\-]?\s*(male|female|m|f)\b|\bgender\s*[:\-]?\s*(male|female|m|f)\b", blob, re.I)
        if m: found["sex"] = (m.group(1) or m.group(2))[0].upper()
    m = re.search(r"(?:patient\s*name|name)\s*[:\-]\s*([A-Za-z][A-Za-z .]{1,40}?)(?=\s+(?:report|age|sex|date|id|no|uhid|ref)\b|\s*$|\s*:)", blob, re.I | re.M)
    if m: found["patient_name"] = m.group(1).strip()

    # ---- numeric analytes
    for key, (pat, excl) in NAMES.items():
        for line in lines:
            if excl and re.search(excl, line, re.I):
                continue
            mm = re.search(pat, line, re.I)
            if not mm:
                continue
            if key == "bp":  # "BP 130/80" -> diastolic
                b = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", line[mm.end():])
                if b: found["bp"] = float(b.group(2)); notes["bp"] = line; break
                continue
            val = _first_number(line[mm.end():])
            if val is None:
                continue
            low = line.lower()
            if key == "sc" and re.search(r"[µμu]mol", low): val = round(val / 88.4, 2)
            if key == "bu" and re.search(r"\bbun\b|urea\s+nitrogen", low): val = round(val * 2.14, 1)
            if key == "bgr" and "mmol" in low: val = round(val * 18, 0)
            if key == "hemo" and val > 30: val = round(val / 10, 1)        # g/L -> g/dL
            if key == "wc" and val < 100: val = val * 1000                  # x10^3/uL
            if key == "rc" and val > 100: val = round(val / 1e6, 2) if val > 10000 else round(val / 1000, 2)
            if key == "sg" and 1000 <= val <= 1050: val = val / 1000
            if key in RANGE and not (RANGE[key][0] <= val <= RANGE[key][1]):
                alt = val / 10 if float(val).is_integer() and val >= 10 else None   # "32" -> 3.2 (decimal point lost)
                if alt is not None and RANGE[key][0] <= alt <= RANGE[key][1]: val = alt
                else: continue
            found[key] = val; notes[key] = line; break

    # ---- urine dipstick qualitative (albumin / sugar)
    for key, pat in (("al", r"(?:urine\s+)?(?:albumin|protein)"), ("su", r"(?:urine\s+)?(?:sugar|glucose)")):
        for line in lines:
            low = line.lower()
            if "serum" in low or "blood" in low or "random" in low or "fasting" in low: continue
            mm = re.search(pat, low)
            if not mm: continue
            rest = low[mm.end():].strip(" :-")
            q = re.match(r"(not detected|nil|negative|neg|absent|trace|present|\+{1,4}|[1-4]\+)", rest)
            if q: found[key] = float(QUAL[q.group(1)]); notes[key] = line; break
            n = re.match(r"([0-5])\b", rest)
            if n and key == "al" and not re.search(r"g/d?l|mg", rest[:12]): found[key] = float(n.group(1)); notes[key] = line; break

    # ---- urine microscopy: pus cells / RBC (first value after the label, not the reference range)
    for key, pat, cut in (("pc", r"pus\s+c[ae]ll?s?|w\.?b\.?c\.?s?\s*\(?urine|leuc?ocytes?\s*\(?urine", 5),
                          ("rbc", r"(?:urine\s+)?(?:r\.?b\.?c\.?s?|red\s+blood\s+cells?)", 2)):
        for line in lines:
            low = line.lower()
            mm = re.search(pat, low)
            if not mm or (key == "rbc" and "hpf" not in low and "urine" not in low): continue
            rest = low[mm.end():].strip(" :-")
            head = rest[:24]
            r = re.match(r"(\d+)\s*[-\u2013:]\s*(\d+)", head)
            if r: found[key] = 1.0 if int(r.group(2)) > cut else 0.0
            elif re.search(r"abnormal|numerous|plenty|many|loaded|\+", head): found[key] = 1.0
            elif re.search(r"normal|nil|absent|occasional|none|not seen", head): found[key] = 0.0
            else: continue
            notes[key] = line; break
    return found, notes


def extract(path):
    texts, method = read_texts(path)
    parsed = [parse_report(t) for t in texts]
    if len(parsed) == 1:
        found, notes = parsed[0]
    else:  # vote per field across OCR variants; ties go to the earliest (best-tuned) variant
        found, notes = {}, {}
        keys = []
        for f, _ in parsed:
            for k in f:
                if k not in keys: keys.append(k)
        for k in keys:
            votes = [(f[k], n.get(k)) for f, n in parsed if k in f]
            vals = [v for v, _ in votes]
            best = max(dict.fromkeys(vals), key=lambda v: vals.count(v))
            found[k] = best
            notes[k] = next(n for v, n in votes if v == best)
    return {"values": found, "lines": notes, "method": method, "chars": sum(len(t) for t in texts)}
