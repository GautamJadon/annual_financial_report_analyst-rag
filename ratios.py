"""Financial statement extraction + ratio calculator.

Numbers are pulled from the balance sheet and P&L by code (no LLM), stored in
SQLite, and every ratio is computed with plain Python formulas.
"""
import json
import re
import sqlite3

import pymupdf

DB_FILE = "financials.db"
NUM = r"(?:\(?[\d,]+\.\d{1,2}\)?|[-–—])"
LINE_RE = re.compile(rf"^(.*?)(?:\s+|^)({NUM})\s+({NUM})\s*$")

# key -> (section or None, [label regexes], mode)   section: NCA CA EQ NCL CL
PATTERNS = {
    # balance sheet
    "total_assets": (None, [r"^total assets$"], "first"),
    "total_equity": (None, [r"^total equity$"], "first"),
    "total_liabilities": (None, [r"^total liabilities$"], "first"),
    "total_equity_and_liabilities": (None, [r"^total equity and liabilities$"], "first"),
    "current_assets": (None, [r"^total current assets"], "first"),
    "current_liabilities": (None, [r"^total current liabilities"], "first"),
    "noncurrent_liabilities": (None, [r"^total non-current liabilities"], "first"),
    "inventories": ("CA", [r"^inventories"], "first"),
    "trade_receivables": ("CA", [r"^trade receivables"], "first"),
    "cash": ("CA", [r"^cash and cash equivalents"], "first"),
    "bank_balances": ("CA", [r"^bank balances"], "first"),
    "current_investments": ("CA", [r"^investments"], "first"),
    "borrowings_nc": ("NCL", [r"^borrowings"], "first"),
    "borrowings_c": ("CL", [r"^borrowings"], "first"),
    "lease_nc": ("NCL", [r"^lease liabilities"], "first"),
    "lease_c": ("CL", [r"^lease liabilities"], "first"),
    "trade_payables": ("CL", [r"^total outstanding dues of micro", r"^total outstanding dues of creditors", r"^trade payables"], "sum"),
    # profit and loss
    "revenue": (None, [r"^revenue from operations"], "first"),
    "other_income": (None, [r"^other income$"], "first"),
    "total_income": (None, [r"^total income"], "first"),
    "cost_of_materials": (None, [r"^cost of materials consumed"], "first"),
    "purchases_stock": (None, [r"^purchases of stock-in-trade"], "first"),
    "changes_inventories": (None, [r"^changes in inventories"], "first"),
    "employee_expense": (None, [r"^employee benefits? expense"], "first"),
    "finance_costs": (None, [r"^finance costs?"], "first"),
    "depreciation": (None, [r"^depreciation and amorti[sz]ation"], "first"),
    "other_expenses": (None, [r"^other expenses$"], "first"),
    "total_expenses": (None, [r"^total expenses"], "first"),
    "pbt": (None, [r"^profit before tax"], "first"),
    "tax_expense": (None, [r"^total tax expenses?"], "first"),
    "pat": (None, [r"^profit for the (year|period)"], "first"),
}
BS_KEYS = ["total_assets", "total_equity", "total_liabilities", "total_equity_and_liabilities", "current_assets",
           "current_liabilities", "noncurrent_liabilities", "inventories", "trade_receivables", "cash",
           "bank_balances", "current_investments", "borrowings_nc", "borrowings_c", "lease_nc", "lease_c",
           "trade_payables"]
PL_KEYS = [k for k in PATTERNS if k not in BS_KEYS]

LABELS = {
    "total_assets": "Total assets", "total_equity": "Total equity", "total_liabilities": "Total liabilities",
    "total_equity_and_liabilities": "Total equity and liabilities", "current_assets": "Total current assets",
    "current_liabilities": "Total current liabilities", "noncurrent_liabilities": "Total non-current liabilities",
    "inventories": "Inventories", "trade_receivables": "Trade receivables", "cash": "Cash and cash equivalents",
    "bank_balances": "Other bank balances", "current_investments": "Current investments",
    "borrowings_nc": "Borrowings (non-current)", "borrowings_c": "Borrowings (current)",
    "lease_nc": "Lease liabilities (non-current)", "lease_c": "Lease liabilities (current)",
    "trade_payables": "Trade payables", "revenue": "Revenue from operations", "other_income": "Other income",
    "total_income": "Total income", "cost_of_materials": "Cost of materials consumed",
    "purchases_stock": "Purchases of stock-in-trade", "changes_inventories": "Changes in inventories",
    "employee_expense": "Employee benefits expense", "finance_costs": "Finance costs",
    "depreciation": "Depreciation and amortisation", "other_expenses": "Other expenses",
    "total_expenses": "Total expenses", "pbt": "Profit before tax", "tax_expense": "Total tax expense",
    "pat": "Profit for the year",
}


# ---------- locating statements ----------
def _regions(page):
    r = page.rect
    if r.width > 1.2 * r.height:  # two-page spread
        mid = r.width / 2
        return [("L", pymupdf.Rect(0, 0, mid, r.height)), ("R", pymupdf.Rect(mid, 0, r.width, r.height))]
    return [("", r)]


def locate_statements(doc):
    """Return {(basis, 'bs'|'pl'): (page_number, side, clip)} using the first match in the document."""
    found = {}
    for i, page in enumerate(doc, 1):
        for side, clip in _regions(page):
            text = page.get_text("text", clip=clip)
            up = text.upper()
            head = up[:600]
            basis = "consolidated" if "CONSOLIDATED" in head else "standalone"
            if "BALANCE SHEET" in head and "TOTAL CURRENT ASSETS" in up:
                found.setdefault((basis, "bs"), (i, side, clip))
            elif "STATEMENT OF PROFIT AND LOSS" in head and "REVENUE FROM OPERATIONS" in up:
                found.setdefault((basis, "pl"), (i, side, clip))
    return found


# ---------- parsing lines ----------
def _to_float(tok):
    tok = tok.strip()
    if tok in "-–—":
        return 0.0
    neg = tok.startswith("(") and tok.endswith(")")
    val = float(tok.strip("()").replace(",", ""))
    return -val if neg else val


def _clean_label(label):
    label = re.sub(r"[ \t]+", " ", label).strip()
    for _ in range(3):
        label = re.sub(r"^(?:\([A-Za-z0-9]{1,4}\)|[IVX]{1,5}|\d+\.)\s+", "", label)
    label = re.sub(r"(?:\s+\d+(?:\s*\([a-z]\))?(?:\s*,\s*\d+)*)+$", "", label)
    return label.strip()


def _section_for(label):
    l = label.lower()
    if l.startswith("non-current assets"):
        return "NCA"
    if l.startswith("current assets"):
        return "CA"
    if l == "equity":
        return "EQ"
    if l.startswith("non-current liabilities"):
        return "NCL"
    if l.startswith("current liabilities"):
        return "CL"
    return None


def parse_lines(text):
    """Turn statement text into [{section, label, cur, prev}]."""
    entries, section, header = [], None, ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            label = _clean_label(line)
            sec = _section_for(label)
            if sec:
                section = sec
            if label:
                header = label
            continue
        label = _clean_label(m.group(1)) or header
        entries.append({"section": section, "label": label,
                        "cur": _to_float(m.group(2)), "prev": _to_float(m.group(3))})
    return entries


def _find(entries, section, patterns, mode):
    hits = []
    for pat in patterns:
        for e in entries:
            if section and e["section"] != section:
                continue
            if re.search(pat, e["label"].lower()):
                hits.append(e)
                if mode == "first":
                    return e["cur"], e["prev"]
                break
    if mode == "sum" and hits:
        # prefer the two sub-lines over the header line when both exist
        subs = [h for h in hits if h["label"].lower().startswith("total outstanding")]
        use = subs or hits[:1]
        return sum(h["cur"] for h in use), sum(h["prev"] for h in use)
    return None


def _units_and_periods(text):
    units = ""
    m = re.search(r"\(All amounts[^)]*\)", text, re.I)
    if m:
        units = m.group(0).strip("()")
    p = re.search(r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", text)
    return units, (p.group(1) if p else "Current"), (p.group(2) if p else "Previous")


def extract_pdf(path):
    """Extract statements from a PDF. Returns {basis: data} (empty if nothing found)."""
    out = {}
    with pymupdf.open(path) as doc:
        found = locate_statements(doc)
        for basis in ("standalone", "consolidated"):
            if (basis, "bs") not in found or (basis, "pl") not in found:
                continue
            metrics, pages, units, cur_l, prev_l = {}, {}, "", "Current", "Previous"
            for kind, keys in (("bs", BS_KEYS), ("pl", PL_KEYS)):
                pno, side, clip = found[(basis, kind)]
                text = doc[pno - 1].get_text("text", clip=clip, sort=True)
                if kind == "bs":
                    units, cur_l, prev_l = _units_and_periods(text)
                entries = parse_lines(text)
                for k in keys:
                    section, pats, mode = PATTERNS[k]
                    res = _find(entries, section, pats, mode)
                    if k == "revenue" and res is None:
                        a, b = _find(entries, None, [r"^sale of goods"], "first") or (None, None)
                        c, d = _find(entries, None, [r"^other operating revenues"], "first") or (0.0, 0.0)
                        res = (a + c, b + d) if a is not None else None
                    if res:
                        metrics[k] = {"cur": round(res[0], 2), "prev": round(res[1], 2), "page": f"{pno}{side}"}
            out[basis] = {"units": units, "cur_label": cur_l, "prev_label": prev_l, "metrics": metrics}
    return out


# ---------- ratios ----------
def _period_values(metrics, period):
    return {k: v.get(period) for k, v in metrics.items() if v.get(period) is not None}


def compute_ratios(m):
    """m: {metric: value} for ONE period. Returns list of (group, name, formula, value, kind)."""
    def g(k, default=None):
        return m.get(k, default)

    def div(a, b):
        return None if a is None or b in (None, 0) else a / b

    cogs = None
    if g("cost_of_materials") is not None:
        cogs = g("cost_of_materials") + g("purchases_stock", 0) + g("changes_inventories", 0)
    ebit = None if g("pbt") is None or g("finance_costs") is None else g("pbt") + g("finance_costs")
    ebitda = None if ebit is None or g("depreciation") is None else ebit + g("depreciation")
    debt = None
    if g("borrowings_nc") is not None or g("borrowings_c") is not None:
        debt = sum(g(k, 0) for k in ("borrowings_nc", "borrowings_c", "lease_nc", "lease_c"))
    cash_like = None if g("cash") is None else g("cash") + g("bank_balances", 0) + g("current_investments", 0)
    quick = None if g("current_assets") is None or g("inventories") is None else g("current_assets") - g("inventories")
    rev, ca, cl = g("revenue"), g("current_assets"), g("current_liabilities")

    return [
        ("Liquidity", "Current ratio", "Current assets / Current liabilities", div(ca, cl), "x"),
        ("Liquidity", "Quick ratio", "(Current assets - Inventories) / Current liabilities", div(quick, cl), "x"),
        ("Liquidity", "Cash ratio", "(Cash + Bank balances + Current investments) / Current liabilities", div(cash_like, cl), "x"),
        ("Leverage", "Debt-to-equity", "(Borrowings + Lease liabilities) / Total equity", div(debt, g("total_equity")), "x"),
        ("Leverage", "Liabilities-to-assets", "Total liabilities / Total assets", div(g("total_liabilities"), g("total_assets")), "x"),
        ("Leverage", "Interest coverage", "EBIT (PBT + Finance costs) / Finance costs", div(ebit, g("finance_costs")), "x"),
        ("Profitability", "Gross margin", "(Revenue - COGS) / Revenue", None if cogs is None else div(None if rev is None else rev - cogs, rev), "%"),
        ("Profitability", "EBITDA margin", "(PBT + Finance costs + Depreciation) / Revenue", div(ebitda, rev), "%"),
        ("Profitability", "Net profit margin", "Profit for the year / Revenue", div(g("pat"), rev), "%"),
        ("Profitability", "Return on equity", "Profit for the year / Closing total equity", div(g("pat"), g("total_equity")), "%"),
        ("Profitability", "Return on assets", "Profit for the year / Total assets", div(g("pat"), g("total_assets")), "%"),
        ("Efficiency", "Asset turnover", "Revenue / Total assets", div(rev, g("total_assets")), "x"),
        ("Efficiency", "Inventory days", "Inventories / COGS x 365", None if cogs is None else _mul(div(g("inventories"), cogs), 365), "d"),
        ("Efficiency", "Receivable days", "Trade receivables / Revenue x 365", _mul(div(g("trade_receivables"), rev), 365), "d"),
        ("Efficiency", "Payable days", "Trade payables / COGS x 365", None if cogs is None else _mul(div(g("trade_payables"), cogs), 365), "d"),
    ]


def _mul(v, k):
    return None if v is None else v * k


def fmt(value, kind):
    if value is None:
        return "n/a"
    if kind == "%":
        return f"{value * 100:.1f}%"
    if kind == "d":
        return f"{value:.0f} days"
    return f"{value:.2f}x"


def sanity_checks(metrics, period="cur"):
    """Cross-checks that catch extraction mistakes. Returns [(description, ok, detail)]."""
    m = _period_values(metrics, period)
    checks = []

    def add(desc, left, right, parts):
        if all(p in m for p in parts):
            a, b = left(m), right(m)
            checks.append((desc, abs(a - b) <= 0.51, f"{a:,.2f} vs {b:,.2f}"))

    add("Total assets = Equity + Liabilities", lambda x: x["total_assets"],
        lambda x: x["total_equity"] + x["total_liabilities"], ["total_assets", "total_equity", "total_liabilities"])
    add("Total liabilities = Non-current + Current", lambda x: x["total_liabilities"],
        lambda x: x["noncurrent_liabilities"] + x["current_liabilities"],
        ["total_liabilities", "noncurrent_liabilities", "current_liabilities"])
    add("Total income = Revenue + Other income", lambda x: x["total_income"],
        lambda x: x["revenue"] + x["other_income"], ["total_income", "revenue", "other_income"])
    exp = ["cost_of_materials", "purchases_stock", "changes_inventories", "employee_expense", "finance_costs",
           "depreciation", "other_expenses", "total_expenses"]
    add("Total expenses = sum of expense lines", lambda x: x["total_expenses"],
        lambda x: sum(x[k] for k in exp[:-1]), exp)
    return checks


# ---------- storage ----------
def _conn():
    con = sqlite3.connect(DB_FILE)
    con.execute("CREATE TABLE IF NOT EXISTS financials (doc TEXT, basis TEXT, data TEXT, PRIMARY KEY (doc, basis))")
    return con


def save(doc, basis, data):
    with _conn() as con:
        con.execute("INSERT OR REPLACE INTO financials VALUES (?,?,?)", (doc, basis, json.dumps(data)))


def load(doc, basis):
    with _conn() as con:
        row = con.execute("SELECT data FROM financials WHERE doc=? AND basis=?", (doc, basis)).fetchone()
    return json.loads(row[0]) if row else None


def list_docs():
    with _conn() as con:
        return [r[0] for r in con.execute("SELECT DISTINCT doc FROM financials ORDER BY doc")]


def bases_for(doc):
    with _conn() as con:
        return [r[0] for r in con.execute("SELECT basis FROM financials WHERE doc=? ORDER BY basis", (doc,))]


def delete(doc):
    with _conn() as con:
        con.execute("DELETE FROM financials WHERE doc=?", (doc,))


# ---------- chat integration ----------
RATIO_WORDS = re.compile(
    r"ratio|margin|\broe\b|\broa\b|turnover|coverage|leverage|debt.to.equity|\bdays\b|liquidity|solvency|profitability|ebitda",
    re.I)


def is_ratio_question(q):
    return bool(RATIO_WORDS.search(q))


def context_block(doc_names):
    """Pre-computed ratios as text for the LLM. Empty string if nothing stored."""
    parts = []
    for doc in doc_names:
        for basis in bases_for(doc):
            data = load(doc, basis)
            if not data:
                continue
            cur = compute_ratios(_period_values(data["metrics"], "cur"))
            prev = compute_ratios(_period_values(data["metrics"], "prev"))
            lines = [f"COMPUTED RATIOS - {doc} ({basis} statements; calculated by code from the balance sheet and "
                     f"P&L; use as-is). Columns: {data['cur_label']} | {data['prev_label']}"]
            for (grp, name, formula, v1, kind), (_, _, _, v2, _) in zip(cur, prev):
                lines.append(f"- {name} = {formula}: {fmt(v1, kind)} | {fmt(v2, kind)}")
            parts.append("\n".join(lines))
    return "\n\n".join(parts)
