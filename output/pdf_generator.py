"""
output/pdf_generator.py — Professional PDF report generation for FinBot.

Produces a multi-page, fully-formatted PDF containing:

  • Cover page  — company name, price, recommendation badge, date
  • Executive Summary — LLM thesis, price targets, bull/bear cases
  • Technical Analysis — 7 indicators, each with:
        formula box  |  source citation  |  rationale  |  current value + signal
  • Fundamental Analysis — 5 metrics with same structure
  • Statistical Analysis — 5 models with prediction tables
  • Analyst Consensus — consensus rating, price targets, headlines
  • Charts — price, technical (RSI/MACD), Monte Carlo fan chart
  • Mathematical Appendix — all 17 formulas with full descriptions

Uses ReportLab (pure Python, no system-level dependencies).
Output: reports/{TICKER}_{DATE}/report.pdf
"""

import os
import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT

# ── Page geometry ─────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN   = 1.8 * cm
USABLE_W = PAGE_W - 2 * MARGIN

# ── Colour palette ────────────────────────────────────────────────────────────
C_NAVY  = colors.HexColor("#1b2a4a")
C_BLUE  = colors.HexColor("#2563eb")
C_TEAL  = colors.HexColor("#0d7377")
C_GREEN = colors.HexColor("#16a34a")
C_RED   = colors.HexColor("#dc2626")
C_AMBER = colors.HexColor("#d97706")
C_GRAY  = colors.HexColor("#6b7280")
C_LGRAY = colors.HexColor("#f1f5f9")
C_MGRAY = colors.HexColor("#e2e8f0")
C_DGRAY = colors.HexColor("#374151")
C_BLACK = colors.HexColor("#111827")
C_WHITE = colors.white
C_FBKG  = colors.HexColor("#eff6ff")   # formula box background (light blue)
C_FBD   = colors.HexColor("#bfdbfe")   # formula box border


# ── Style factory ─────────────────────────────────────────────────────────────

def _S(name: str, **kw) -> ParagraphStyle:
    """Shorthand ParagraphStyle constructor."""
    return ParagraphStyle(name, **kw)


# All paragraph styles used across the PDF
ST = {
    # Cover page
    "cv_title":  _S("cv_t",  fontName="Helvetica-Bold",    fontSize=30, textColor=C_NAVY,  alignment=TA_CENTER, spaceAfter=4),
    "cv_sub":    _S("cv_s",  fontName="Helvetica",          fontSize=14, textColor=C_BLUE,  alignment=TA_CENTER, spaceAfter=4),
    "cv_price":  _S("cv_p",  fontName="Helvetica-Bold",    fontSize=44, textColor=C_GREEN, alignment=TA_CENTER, spaceAfter=6),
    "cv_meta":   _S("cv_m",  fontName="Helvetica",          fontSize=10, textColor=C_GRAY,  alignment=TA_CENTER, spaceAfter=3),
    # Section headings
    "h2":        _S("h2",    fontName="Helvetica-Bold",    fontSize=11, textColor=C_NAVY,  spaceBefore=8,  spaceAfter=3),
    "h3":        _S("h3",    fontName="Helvetica-Bold",    fontSize=10, textColor=C_TEAL,  spaceBefore=6,  spaceAfter=2),
    # Body text
    "body":      _S("body",  fontName="Helvetica",          fontSize=9,  textColor=C_DGRAY, alignment=TA_JUSTIFY, spaceAfter=3, leading=14),
    "body_c":    _S("bodyc", fontName="Helvetica",          fontSize=9,  textColor=C_DGRAY, alignment=TA_CENTER,  spaceAfter=2, leading=13),
    "bold":      _S("bold",  fontName="Helvetica-Bold",    fontSize=9,  textColor=C_BLACK, spaceAfter=2),
    # Formula / code
    "formula":   _S("fml",   fontName="Courier",            fontSize=8,  textColor=C_NAVY,  leading=13, spaceAfter=2),
    # Small / caption
    "small":     _S("sm",    fontName="Helvetica",          fontSize=8,  textColor=C_GRAY,  spaceAfter=2),
    "smital":    _S("smi",   fontName="Helvetica-Oblique",  fontSize=8,  textColor=C_GRAY,  spaceAfter=1),
    # Disclaimer
    "disclaim":  _S("dis",   fontName="Helvetica-Oblique",  fontSize=7,  textColor=C_GRAY,  alignment=TA_CENTER, leading=10),
}


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _sp(cm_val: float = 0.3) -> Spacer:
    return Spacer(1, cm_val * cm)


def _hr() -> HRFlowable:
    return HRFlowable(width="100%", thickness=0.4, color=C_MGRAY, spaceAfter=3)


def _hex(c) -> str:
    """ReportLab color → CSS hex string (used inside Paragraph markup)."""
    return f"#{int(c.red * 255):02x}{int(c.green * 255):02x}{int(c.blue * 255):02x}"


def _signal_hex(signal: str) -> str:
    """Choose a colour hex based on the signal string content."""
    s = str(signal).upper()
    if any(w in s for w in ("BUY", "UNDERVALUED", "BULLISH", "OVERSOLD",
                             "UPWARD", "GOLDEN", "GOOD", "EXCELLENT")):
        return _hex(C_GREEN)
    if any(w in s for w in ("SELL", "OVERVALUED", "BEARISH", "OVERBOUGHT",
                             "DOWNWARD", "DEATH", "EXPENSIVE", "NEGATIVE")):
        return _hex(C_RED)
    if any(w in s for w in ("HOLD", "NEUTRAL", "FAIR", "MODERATE",
                             "DEFENSIVE", "MARKET-LIKE", "BELOW RISK")):
        return _hex(C_AMBER)
    return _hex(C_GRAY)


def _p(text: str, style_key: str = "body") -> Paragraph:
    """Shorthand Paragraph constructor."""
    return Paragraph(text, ST[style_key])


def _fmt(val, prefix: str = "", suffix: str = "", null: str = "N/A") -> str:
    """Format a possibly-None value for display."""
    if val is None:
        return null
    try:
        return f"{prefix}{float(val):.2f}{suffix}"
    except (TypeError, ValueError):
        return str(val)


# ── Mid-level building blocks ─────────────────────────────────────────────────

def _section_bar(title: str) -> Table:
    """Full-width dark navy section header bar."""
    cell_style = ParagraphStyle(
        "_sb", fontName="Helvetica-Bold", fontSize=12,
        textColor=C_WHITE, alignment=TA_LEFT, leading=18
    )
    return Table(
        [[Paragraph(title, cell_style)]],
        colWidths=[USABLE_W],
        style=TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), C_NAVY),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ]),
    )


def _formula_box(lines: list) -> Table:
    """
    Light-blue styled box for displaying mathematical formulas.
    Each string in *lines* becomes one line of Courier-font content.
    """
    content = "<br/>".join(lines)
    return Table(
        [[Paragraph(content, ST["formula"])]],
        colWidths=[USABLE_W],
        style=TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), C_FBKG),
            ("BOX",           (0, 0), (-1, -1), 0.75, C_FBD),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]),
    )


def _indicator_header(name: str, value_str: str) -> Table:
    """Two-column row: indicator name (left) + current value (right)."""
    left_st  = ParagraphStyle("_ih_l", fontName="Helvetica-Bold", fontSize=10, textColor=C_NAVY)
    right_st = ParagraphStyle("_ih_r", fontName="Helvetica-Bold", fontSize=10,
                               textColor=C_TEAL, alignment=TA_RIGHT)
    return Table(
        [[Paragraph(name, left_st), Paragraph(value_str, right_st)]],
        colWidths=[USABLE_W * 0.65, USABLE_W * 0.35],
        style=TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), C_LGRAY),
            ("LINEBELOW",     (0, 0), (-1, -1), 1.0, C_BLUE),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]),
    )


def _indicator_block(
    name: str,
    current_value: str,
    formula_lines: list,
    source: str,
    rationale: str,
    analysis: str,
    signal: str,
) -> list:
    """
    Assemble one complete indicator block (wrapped in KeepTogether so it
    won't be split across pages) containing:
      • Name + current value header bar
      • Mathematical formula box
      • Source citation
      • Rationale (why this metric is used)
      • Analysis (what the current reading means for THIS stock)
      • Signal badge
    """
    sig_col = _signal_hex(signal)
    elements = [
        _indicator_header(name, f"Current: {current_value}"),
        _sp(0.15),
        _formula_box(formula_lines),
        _sp(0.12),
        Paragraph(f"<i>Source: {source}</i>", ST["smital"]),
        _sp(0.08),
        Paragraph(f"<b>Why this metric:</b> {rationale}", ST["body"]),
        _sp(0.08),
        Paragraph(f"<b>Analysis:</b> {analysis}", ST["body"]),
        _sp(0.08),
        Paragraph(
            f'<b>Signal: </b><font color="{sig_col}"><b>{signal}</b></font>',
            ST["body"],
        ),
        _sp(0.2),
        _hr(),
        _sp(0.1),
    ]
    return [KeepTogether(elements)]


def _generic_table(headers: list, rows: list, col_widths: list) -> Table:
    """
    General-purpose styled data table with navy header row and
    alternating light/white row backgrounds.
    """
    hdr_st = ParagraphStyle("_gt_h", fontName="Helvetica-Bold",
                             fontSize=8, textColor=C_WHITE, alignment=TA_CENTER)
    cel_st = ParagraphStyle("_gt_c", fontName="Helvetica",
                             fontSize=8, textColor=C_DGRAY, alignment=TA_CENTER)
    data = (
        [[Paragraph(str(h), hdr_st) for h in headers]]
        + [[Paragraph(str(c), cel_st) for c in row] for row in rows]
    )
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1,  0), C_NAVY),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_MGRAY),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def _embed_chart(path: str, label: str) -> list:
    """
    Return a list of flowables embedding a chart PNG, or a placeholder
    paragraph if the file doesn't exist.
    """
    if os.path.exists(path):
        img_w = USABLE_W
        # Preserve aspect ratio based on figsize used when generating the chart
        aspect = {"price_chart": 8/14, "technical_chart": 10/14, "monte_carlo_chart": 7/14}
        key = os.path.splitext(os.path.basename(path))[0]
        ratio = aspect.get(key, 0.6)
        img_h = img_w * ratio
        return [
            _p(f"<b>{label}</b>", "h3"),
            Image(path, width=img_w, height=img_h, kind="proportional"),
            _sp(0.3),
        ]
    return [_p(f"[Chart not available: {label}]", "small")]


# ── Section builders ──────────────────────────────────────────────────────────

def _build_cover(ticker: str, info: dict, llm: dict) -> list:
    """Generate the cover page flowables."""
    name    = info.get("shortName", ticker)
    price   = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
    sector  = info.get("sector", "N/A")
    mcap    = info.get("marketCap")
    mcap_s  = f"${mcap:,.0f}" if mcap else "N/A"
    hi52    = info.get("fiftyTwoWeekHigh", "N/A")
    lo52    = info.get("fiftyTwoWeekLow",  "N/A")
    date_s  = datetime.date.today().isoformat()

    # Pull recommendation/confidence from the first available provider
    rec, conf = "", ""
    if llm.get("llm_available"):
        first_result = next(iter(llm.get("providers", {}).values()), {})
        rec  = first_result.get("recommendation", "")
        conf = first_result.get("confidence", "")

    # Recommendation badge colour
    if "BUY" in str(rec).upper():
        rec_col, badge_bg = _hex(C_WHITE), C_GREEN
    elif "SELL" in str(rec).upper():
        rec_col, badge_bg = _hex(C_WHITE), C_RED
    elif rec:
        rec_col, badge_bg = _hex(C_WHITE), C_AMBER
    else:
        rec_col, badge_bg = _hex(C_GRAY), C_LGRAY

    price_str = f"${price}" if isinstance(price, (int, float)) else str(price)

    story = [
        _sp(2.0),
        # FinBot brand line
        Paragraph("FinBot", _S("_cv_brand", fontName="Helvetica-Bold", fontSize=16,
                               textColor=C_TEAL, alignment=TA_CENTER, spaceAfter=2)),
        Paragraph("AI-Powered Quantitative Stock Analysis", _S("_cv_bsub", fontName="Helvetica",
                  fontSize=10, textColor=C_GRAY, alignment=TA_CENTER, spaceAfter=16)),
        _hr(),
        _sp(0.8),
        # Company name + ticker
        Paragraph(name, ST["cv_title"]),
        Paragraph(ticker, ST["cv_sub"]),
        _sp(0.5),
        # Current price (large, green)
        Paragraph(price_str, ST["cv_price"]),
        _sp(0.3),
    ]

    # Recommendation badge (only if LLM ran)
    if rec:
        badge_st = ParagraphStyle(
            "_badge", fontName="Helvetica-Bold", fontSize=18,
            textColor=C_WHITE, alignment=TA_CENTER, leading=24,
        )
        badge_tbl = Table(
            [[Paragraph(f"{rec}  ·  Confidence: {conf}", badge_st)]],
            colWidths=[USABLE_W * 0.6],
            style=TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), badge_bg),
                ("TOPPADDING",    (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING",   (0, 0), (-1, -1), 16),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 16),
            ]),
            hAlign="CENTER",
        )
        story += [badge_tbl, _sp(0.5)]

    # Company metadata grid
    meta_rows = [
        ["Sector",       sector,         "52W High", f"${hi52}"],
        ["Market Cap",   mcap_s,         "52W Low",  f"${lo52}"],
        ["Report Date",  date_s,         "Ticker",   ticker],
    ]
    meta_st = ParagraphStyle("_ms", fontName="Helvetica",     fontSize=9, textColor=C_DGRAY, alignment=TA_CENTER)
    metab_st = ParagraphStyle("_mb", fontName="Helvetica-Bold", fontSize=9, textColor=C_NAVY,  alignment=TA_CENTER)
    meta_data = [
        [Paragraph(label, metab_st), Paragraph(val, meta_st),
         Paragraph(label2, metab_st), Paragraph(val2, meta_st)]
        for label, val, label2, val2 in meta_rows
    ]
    meta_tbl = Table(
        meta_data,
        colWidths=[USABLE_W * 0.18, USABLE_W * 0.32, USABLE_W * 0.18, USABLE_W * 0.32],
        style=TableStyle([
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_LGRAY, C_WHITE]),
            ("GRID",           (0, 0), (-1, -1), 0.3, C_MGRAY),
            ("TOPPADDING",     (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
            ("LEFTPADDING",    (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",   (0, 0), (-1, -1), 8),
        ]),
    )
    story += [_sp(0.6), meta_tbl]
    return story


def _build_executive_summary(llm: dict, analyst_data: dict, info: dict) -> list:
    """
    Executive summary section: LLM thesis, price targets, bull/bear cases,
    key risks, catalysts, and analyst consensus overview.
    """
    story = [_section_bar("EXECUTIVE SUMMARY"), _sp(0.3)]

    # LLM narrative
    if llm.get("llm_available"):
        providers = llm.get("providers", {})
        for provider_name, result in providers.items():
            # Provider header
            story.append(_p(f"<b>{provider_name}</b>", "h2"))
            story.append(_hr())

            rec    = result.get("recommendation", "N/A")
            conf   = result.get("confidence", "N/A")
            summ   = result.get("summary", "")
            tech_v = result.get("technical_verdict", "")
            fund_v = result.get("fundamental_verdict", "")
            stat_v = result.get("statistical_verdict", "")
            agents = result.get("agents", {})

            # ── Agent panels ──────────────────────────────────────── #
            agent_defs = [
                ("mathematician", "Agent 1 — Mathematician",  C_BLUE),
                ("quant",         "Agent 2 — Quant Analyst",  C_TEAL),
                ("market",        "Agent 3 — Market Analyst", C_NAVY),
                ("shareholder",   "Agent 4 — Shareholder",    C_GREEN),
            ]
            for ag_key, title, colour in agent_defs:
                ag = agents.get(ag_key, {})
                findings = ag.get("findings", [])
                verdict  = ag.get("verdict", "")
                if not findings and not verdict:
                    continue
                story.append(_p(f"<b>{title}</b>", "h3"))
                # Extra metadata
                if ag_key == "mathematician":
                    bias = ag.get("statistical_bias")
                    if bias and bias != "N/A":
                        story.append(Paragraph(f"Statistical Bias: {bias}", ST["body"]))
                    formulas = ag.get("formulas_used", [])
                    if formulas:
                        story.append(Paragraph(
                            "Formulas: " + ", ".join(f.get("name", "") for f in formulas[:4]),
                            ST["body"],
                        ))
                elif ag_key == "quant":
                    ep = ag.get("entry_price"); ex = ag.get("exit_price"); sl = ag.get("stop_loss")
                    lvl = ag.get("key_levels", [])
                    meta = []
                    if ep: meta.append(f"Entry: ${ep}")
                    if ex: meta.append(f"Exit: ${ex}")
                    if sl: meta.append(f"Stop: ${sl}")
                    if meta:
                        story.append(Paragraph("  ·  ".join(meta), ST["bold"]))
                    if lvl:
                        story.append(Paragraph(f"Key Levels: {' | '.join(lvl)}", ST["body"]))
                elif ag_key == "market":
                    meta = []
                    for mk, mlbl in [("competitive_position","Position"),("sector_outlook","Sector"),
                                      ("profitability_quality","Profitability"),("analyst_conviction","Conv.")]:
                        v = ag.get(mk)
                        if v and v != "N/A": meta.append(f"{mlbl}: {v}")
                    if meta: story.append(Paragraph("  ·  ".join(meta), ST["body"]))
                elif ag_key == "shareholder":
                    meta = []
                    for mk, mlbl in [("management_rating","Mgmt"),("earnings_consistency","Earnings")]:
                        v = ag.get(mk)
                        if v and v != "N/A": meta.append(f"{mlbl}: {v}")
                    ceo = ag.get("ceo_assessment","")
                    if meta: story.append(Paragraph("  ·  ".join(meta), ST["body"]))
                    if ceo: story.append(Paragraph(f"CEO: {ceo}", ST["body"]))
                    alt   = ag.get("alternative_ticker") or result.get("alternative_pick")
                    alt_r = ag.get("alternative_reason") or result.get("alternative_reason")
                    if alt:
                        story.append(Paragraph(
                            f'<b>Alternative Pick:</b> <font color="{_hex(C_AMBER)}">{alt}</font>'
                            f' — {alt_r or ""}', ST["body"]))
                for f in findings:
                    story.append(Paragraph(
                        f'<font color="{_hex(colour)}">•</font>  {f}', ST["body"]))
                if verdict and verdict != "Unavailable":
                    story.append(Paragraph(f"<i>{verdict}</i>", ST["smital"]))
                story.append(_sp(0.2))

            # ── Judge verdict ─────────────────────────────────────── #
            story.append(_hr())
            sig_col = _signal_hex(rec)
            story.append(Paragraph(
                f'<b>THE JUDGE — Recommendation: </b><font color="{sig_col}"><b>{rec}</b></font>'
                f'  ·  <b>Confidence:</b> {conf}',
                ST["h2"]
            ))
            story.append(_sp(0.1))
            story.append(Paragraph(f"<b>Investment Thesis:</b> {summ}", ST["body"]))
            story.append(_sp(0.15))

            # Trading levels
            ep = result.get("entry_price"); ex = result.get("exit_price"); sl = result.get("stop_loss")
            direction = result.get("trade_direction") or ""
            if ep or ex or sl:
                lvl_rows = []
                if ep: lvl_rows.append(["Entry",     f"${ep}"])
                if ex: lvl_rows.append(["Exit",      f"${ex}"])
                if sl: lvl_rows.append(["Stop Loss", f"${sl}"])
                dir_note = f"  ({direction})" if direction else ""
                story.append(_p(f"<b>Trading Levels</b>{dir_note}", "h3"))
                cw = [USABLE_W * 0.4, USABLE_W * 0.6]
                story.append(_generic_table(["Level", "Price"], lvl_rows, cw))
                story.append(_sp(0.15))

            story.append(Paragraph(f"<b>Technical Verdict:</b> {tech_v}", ST["body"]))
            story.append(Paragraph(f"<b>Fundamental Verdict:</b> {fund_v}", ST["body"]))
            story.append(Paragraph(f"<b>Statistical Verdict:</b> {stat_v}", ST["body"]))
            story.append(_sp(0.2))

            # Price targets
            targets = result.get("target_prices", {})
            if targets:
                label_map = {
                    "1_week":    "1 Week",
                    "2_weeks":   "2 Weeks",
                    "3_weeks":   "3 Weeks",
                    "1_month":   "1 Month",
                    "3_months":  "3 Months",
                    "6_months":  "6 Months",
                    "9_months":  "9 Months",
                    "12_months": "12 Months",
                }
                price = info.get("currentPrice") or 0
                rows = []
                for hkey, lbl in label_map.items():
                    tgt = targets.get(hkey)
                    # Guard against empty string or non-numeric values from the LLM
                    try:
                        tgt_f = float(tgt)
                    except (TypeError, ValueError):
                        tgt_f = None
                    if tgt_f is not None and tgt_f > 0:
                        chg = (tgt_f - float(price)) / float(price) * 100 if price else 0
                        sign = "+" if chg >= 0 else ""
                        rows.append([lbl, f"${tgt_f:.2f}", f"{sign}{chg:.1f}%"])
                if rows:
                    story.append(_p("<b>AI Price Targets</b>", "h3"))
                    cw = [USABLE_W * 0.33] * 3
                    story.append(_generic_table(["Horizon", "Target Price", "Expected Change"], rows, cw))
                    story.append(_sp(0.25))

            # Bull / Bear cases
            bull  = result.get("key_bull_case", [])
            bear  = result.get("key_bear_case", [])
            risks = result.get("key_risks", [])
            cats  = result.get("catalysts", [])

            if bull or bear:
                left_cells, right_cells = [], []
                if bull:
                    left_cells.append(Paragraph("<b>Bull Case</b>", ST["bold"]))
                    for b in bull:
                        left_cells.append(
                            Paragraph(f'<font color="{_hex(C_GREEN)}">+</font>  {b}', ST["body"])
                        )
                if bear:
                    right_cells.append(Paragraph("<b>Bear Case</b>", ST["bold"]))
                    for b in bear:
                        right_cells.append(
                            Paragraph(f'<font color="{_hex(C_RED)}">-</font>  {b}', ST["body"])
                        )

                max_len = max(len(left_cells), len(right_cells))
                left_cells  += [Paragraph("", ST["small"])] * (max_len - len(left_cells))
                right_cells += [Paragraph("", ST["small"])] * (max_len - len(right_cells))

                bb_tbl = Table(
                    [[l, r] for l, r in zip(left_cells, right_cells)],
                    colWidths=[USABLE_W * 0.5, USABLE_W * 0.5],
                    style=TableStyle([
                        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ]),
                )
                story.append(bb_tbl)
                story.append(_sp(0.2))

            if risks:
                story.append(_p("<b>Key Risks</b>", "h3"))
                for r in risks:
                    story.append(Paragraph(f'<font color="{_hex(C_RED)}">  !</font>  {r}', ST["body"]))
                story.append(_sp(0.15))

            if cats:
                story.append(_p("<b>Catalysts</b>", "h3"))
                for c in cats:
                    story.append(Paragraph(f'<font color="{_hex(C_AMBER)}">  *</font>  {c}', ST["body"]))
                story.append(_sp(0.2))

            if result.get("alternative_pick"):
                story.append(Paragraph(
                    f'<b>Alternative Pick:</b> <font color="{_hex(C_AMBER)}">'
                    f'{result["alternative_pick"]}</font> — {result.get("alternative_reason","")}',
                    ST["body"]
                ))
                story.append(_sp(0.15))

            story.append(_sp(0.3))   # gap between providers
    else:
        story.append(Paragraph(
            f"<i>LLM analysis not available: {llm.get('summary', '')}</i>",
            ST["smital"]
        ))
        story.append(_sp(0.3))

    # Analyst consensus summary
    rec_key = analyst_data.get("recommendation_key", "N/A").upper()
    pt      = analyst_data.get("price_target") or {}
    story.append(_hr())
    story.append(_p("<b>Analyst Consensus (Yahoo Finance)</b>", "h3"))
    sig_col = _signal_hex(rec_key)
    story.append(Paragraph(
        f'Consensus: <font color="{sig_col}"><b>{rec_key}</b></font>  '
        f'·  Mean Target: ${pt.get("mean","N/A")}  '
        f'·  Low: ${pt.get("low","N/A")}  '
        f'·  High: ${pt.get("high","N/A")}',
        ST["body"],
    ))

    news = analyst_data.get("news", [])
    if news:
        story.append(_sp(0.1))
        story.append(_p("<b>Recent Headlines</b>", "bold"))
        for item in news[:5]:
            story.append(Paragraph(
                f'<font color="{_hex(C_BLUE)}">•</font>  {item.get("title","")} '
                f'<font color="{_hex(C_GRAY)}">({item.get("publisher","")})</font>',
                ST["body"],
            ))

    # ---- Analyst upgrade/downgrade actions table ------------------- #
    recs_list = analyst_data.get("recommendations", [])
    if recs_list:
        story.append(_sp(0.2))
        story.append(_p("<b>Analyst Upgrade / Downgrade History</b>  (most recent first)", "h3"))

        # Sentiment summary line
        recent_10 = recs_list[:10]
        n_up   = sum(1 for r in recent_10 if "Upgrade"    in r.get("action", ""))
        n_down = sum(1 for r in recent_10 if "Downgrade"  in r.get("action", ""))
        n_init = sum(1 for r in recent_10 if "Initiation" in r.get("action", ""))
        if n_up > n_down:
            sent_col, sent_txt = _hex(C_GREEN), "Net Positive"
        elif n_down > n_up:
            sent_col, sent_txt = _hex(C_RED),   "Net Negative"
        else:
            sent_col, sent_txt = _hex(C_AMBER),  "Mixed / Neutral"

        story.append(Paragraph(
            f'Last 10 analyst actions: '
            f'<font color="{_hex(C_GREEN)}"><b>{n_up} upgrade(s)</b></font>  ·  '
            f'<font color="{_hex(C_RED)}"><b>{n_down} downgrade(s)</b></font>  ·  '
            f'<font color="{_hex(C_BLUE)}">{n_init} new initiation(s)</font>  —  '
            f'Sentiment: <font color="{sent_col}"><b>{sent_txt}</b></font>',
            ST["body"],
        ))
        story.append(_sp(0.1))

        # Action colour function
        def _action_col(action: str) -> str:
            if "Upgrade" in action or "Initiation" in action:
                return _hex(C_GREEN)
            if "Downgrade" in action:
                return _hex(C_RED)
            return _hex(C_AMBER)

        act_st = ParagraphStyle("_act", fontName="Helvetica", fontSize=8,
                                textColor=C_DGRAY, alignment=TA_CENTER)
        def _act_cell(r):
            col = _action_col(r.get("action", ""))
            return Paragraph(
                f'<font color="{col}"><b>{r.get("action","")}</b></font>', act_st
            )

        hdr_st = ParagraphStyle("_agt_h", fontName="Helvetica-Bold", fontSize=8,
                                textColor=C_WHITE, alignment=TA_CENTER)
        cel_st = ParagraphStyle("_agt_c", fontName="Helvetica",      fontSize=8,
                                textColor=C_DGRAY, alignment=TA_CENTER)

        cw = [USABLE_W * w for w in [0.14, 0.33, 0.16, 0.185, 0.185]]
        data = [[Paragraph(h, hdr_st) for h in ["Date", "Firm", "Action", "From", "To"]]]
        for r in recs_list[:20]:
            data.append([
                Paragraph(r.get("date", ""),                           cel_st),
                Paragraph(r.get("firm", ""),                           cel_st),
                _act_cell(r),
                Paragraph(r.get("from_grade", "") or "—",              cel_st),
                Paragraph(r.get("to_grade",   "") or "N/A",            cel_st),
            ])

        tbl = Table(data, colWidths=cw)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1,  0), C_NAVY),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
            ("GRID",          (0, 0), (-1, -1), 0.3, C_MGRAY),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(tbl)
        story.append(_sp(0.2))

        # ---- Legend box -------------------------------------------- #
        legend_title_st = ParagraphStyle(
            "_lt", fontName="Helvetica-Bold", fontSize=8, textColor=C_NAVY, spaceAfter=4
        )
        legend_body_st = ParagraphStyle(
            "_lb", fontName="Helvetica", fontSize=7.5, textColor=C_DGRAY, leading=12, spaceAfter=2
        )
        legend_bold_st = ParagraphStyle(
            "_lbold", fontName="Helvetica-Bold", fontSize=7.5, textColor=C_NAVY, spaceAfter=1
        )

        def _lrow(term, meaning, term_col=None):
            col = term_col or _hex(C_DGRAY)
            return [
                Paragraph(f'<font color="{col}"><b>{term}</b></font>', legend_body_st),
                Paragraph(meaning, legend_body_st),
            ]

        legend_data = [
            # Header row
            [Paragraph("ACTIONS", legend_bold_st), Paragraph("", legend_bold_st)],
            _lrow("Upgrade",    "Analyst raises the stock's rating (e.g. Hold → Buy). Bullish signal.",    _hex(C_GREEN)),
            _lrow("Downgrade",  "Analyst cuts the stock's rating (e.g. Buy → Neutral). Bearish signal.",   _hex(C_RED)),
            _lrow("Initiation", "First-ever coverage by this firm — no prior grade exists.",               _hex(C_BLUE)),
            _lrow("Maintained", "Analyst reaffirms the existing rating without any change.",               _hex(C_AMBER)),
            _lrow("Reiterated", "Same as Maintained — a formal restatement of the prior rating.",          _hex(C_AMBER)),
            # Spacer row
            [Paragraph("", legend_body_st), Paragraph("", legend_body_st)],
            # Grades header
            [Paragraph("GRADE SCALE", legend_bold_st), Paragraph("(terminology varies by brokerage)", legend_body_st)],
            _lrow("Strong Buy · Conviction Buy · Top Pick",          "Highest conviction bullish call.",                              _hex(C_GREEN)),
            _lrow("Buy · Outperform · Overweight · Positive",        "Bullish — expected to outperform the market.",                  _hex(C_GREEN)),
            _lrow("Hold · Neutral · Market Perform",
                  "Neutral — returns expected roughly in line with the broader market.", _hex(C_AMBER)),
            _lrow("Equal Weight",
                  "Same as Hold/Neutral. Used by Morgan Stanley, Barclays and others.",  _hex(C_AMBER)),
            _lrow("Sector Perform",
                  "The stock is expected to perform IN LINE with its sector peers — "
                  "neither outperform nor underperform the sector average. "
                  "Used by RBC, CIBC, and similar brokerages. "
                  "Equivalent to Hold / Neutral / Market Perform.",                     _hex(C_AMBER)),
            _lrow("Underperform · Underweight · Sector Underperform","Mild bearish — expected to lag the market.", _hex(C_RED)),
            _lrow("Sell · Strong Sell",                              "Bearish — analyst expects a material price decline.",           _hex(C_RED)),
            # Spacer
            [Paragraph("", legend_body_st), Paragraph("", legend_body_st)],
            # Columns header
            [Paragraph("COLUMNS", legend_bold_st), Paragraph("", legend_bold_st)],
            _lrow("From",     "The rating held by the analyst before today's action."),
            _lrow("To Grade", "The new rating after today's action — this is the current live rating."),
        ]

        leg_tbl = Table(
            legend_data,
            colWidths=[USABLE_W * 0.38, USABLE_W * 0.62],
        )
        leg_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), C_FBKG),
            ("BOX",           (0, 0), (-1, -1), 0.5,  C_FBD),
            ("GRID",          (0, 0), (-1, -1), 0.2,  C_MGRAY),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(Paragraph("Legend — Actions &amp; Grade Terminology", legend_title_st))
        story.append(leg_tbl)

    return story


def _build_technical_section(technical: dict) -> list:
    """
    Technical Analysis section: 7 indicators, each with a complete
    formula derivation block, rationale, and current reading.
    """
    latest  = technical.get("latest",      {})
    signals = technical.get("signals",     {})
    fibs    = technical.get("fibonacci",   {})
    pivots  = technical.get("pivot_levels", {})

    def v(key):  return _fmt(latest.get(key), null="N/A")
    def sig(key): return signals.get(key, "N/A — insufficient history")

    story = [_section_bar("TECHNICAL ANALYSIS"), _sp(0.3)]

    intro = (
        "Technical analysis studies historical price and volume data to identify "
        "patterns and momentum signals. All 7 indicators below are computed from "
        "the 2-year daily OHLCV history fetched from Yahoo Finance. Every formula "
        "is implemented natively in analysis/technical.py with no external indicator "
        "libraries, making the calculations fully auditable."
    )
    story.append(_p(intro, "body"))
    story.append(_sp(0.2))

    # ── 1. SMA ────────────────────────────────────────────────────────────────
    story += _indicator_block(
        name="1. Simple Moving Average (SMA)",
        current_value=f"SMA-20={v('SMA_20')}  SMA-50={v('SMA_50')}  SMA-200={v('SMA_200')}",
        formula_lines=[
            "SMA_n = (1/n) x [P(T-n+1) + P(T-n+2) + ... + P(T)]",
            "",
            "Where:  n   = window length (20, 50, or 200 bars)",
            "        P_t = closing price on day t",
            "        T   = today",
        ],
        source="Dow Theory / Charles H. Dow (late 1800s); Murphy, J. 'Technical Analysis of the Financial Markets' (1999), Ch. 9.",
        rationale=(
            "The Simple Moving Average smooths daily price noise to reveal the underlying "
            "trend direction. SMA-20 captures the 1-month trend, SMA-50 the 2-month "
            "trend, and SMA-200 the annual structural trend. When SMA-50 crosses above "
            "SMA-200 (Golden Cross) it signals a long-term bullish regime change; when "
            "it crosses below (Death Cross) it signals a bearish regime change. Price "
            "trading above SMA-200 confirms a healthy primary uptrend."
        ),
        analysis=(
            f"SMA-20 = {v('SMA_20')}, SMA-50 = {v('SMA_50')}, SMA-200 = {v('SMA_200')}. "
            f"{sig('MA_Trend')}"
        ),
        signal=sig("MA_Trend"),
    )

    # ── 2. EMA ────────────────────────────────────────────────────────────────
    story += _indicator_block(
        name="2. Exponential Moving Average (EMA)",
        current_value=f"EMA-12={v('EMA_12')}  EMA-26={v('EMA_26')}",
        formula_lines=[
            "EMA_t = alpha x P_t + (1 - alpha) x EMA_(t-1)",
            "",
            "Where:  alpha = 2 / (n + 1)   (smoothing factor)",
            "        EMA_0 = P_0            (initialised at first price)",
            "        n     = window length (12 or 26 bars)",
        ],
        source="Appel, G. 'Technical Analysis: Power Tools for Active Investors' (2005).",
        rationale=(
            "Unlike the SMA, the EMA applies exponentially higher weight to more recent "
            "prices, making it more responsive to new information. The alpha factor ensures "
            "the most recent price has the greatest influence while older prices decay "
            "geometrically. EMA-12 and EMA-26 are the primary inputs to the MACD calculation "
            "and together define the short to medium-term price momentum."
        ),
        analysis=(
            f"EMA-12 = {v('EMA_12')}, EMA-26 = {v('EMA_26')}. "
            + ("EMA-12 above EMA-26 signals short-term bullish momentum."
               if (latest.get("EMA_12") or 0) > (latest.get("EMA_26") or 0)
               else "EMA-12 below EMA-26 signals short-term bearish momentum.")
        ),
        signal="BULLISH — EMA-12 > EMA-26"
               if (latest.get("EMA_12") or 0) > (latest.get("EMA_26") or 0)
               else "BEARISH — EMA-12 < EMA-26",
    )

    # ── 3. RSI ────────────────────────────────────────────────────────────────
    rsi_val = latest.get("RSI")
    story += _indicator_block(
        name="3. Relative Strength Index (RSI-14)",
        current_value=v("RSI"),
        formula_lines=[
            "Step 1:  RS  = avg_gain(14) / avg_loss(14)",
            "Step 2:  RSI = 100 - ( 100 / (1 + RS) )",
            "",
            "Where:  avg_gain(14) = Wilder EMA of positive day returns over 14 periods",
            "        avg_loss(14) = Wilder EMA of negative day returns over 14 periods",
            "        Wilder smoothing: EMA with alpha = 1/14",
        ],
        source="Wilder, J.W. 'New Concepts in Technical Trading Systems' (1978), pp. 63-70.",
        rationale=(
            "RSI measures the speed and magnitude of recent price changes on a bounded "
            "0-100 scale, making it directly comparable across different stocks and price "
            "levels. Wilder defined RSI > 70 as 'overbought' — the recent rally has been "
            "too strong and a pullback/consolidation is likely. RSI < 30 is 'oversold' — "
            "selling has been excessive and a bounce is probable. The 50 midline acts as a "
            "trend-strength confirmation: sustained RSI above 50 = bullish momentum."
        ),
        analysis=(
            f"Current RSI = {v('RSI')}. "
            + (f"Above 70: momentum is overextended — high probability of near-term "
               f"consolidation or pullback." if rsi_val and rsi_val > 70
               else f"Below 30: heavily oversold — contrarian buy zone, watch for reversal "
               f"signals." if rsi_val and rsi_val < 30
               else f"Between 30-70: neutral momentum zone with no extreme reading.")
        ),
        signal=sig("RSI"),
    )

    # ── 4. MACD ───────────────────────────────────────────────────────────────
    story += _indicator_block(
        name="4. MACD + Signal Line + Histogram",
        current_value=f"MACD={v('MACD_Line')}  Signal={v('MACD_Signal')}  Hist={v('MACD_Hist')}",
        formula_lines=[
            "MACD_line   = EMA_12(Close) - EMA_26(Close)",
            "Signal_line = EMA_9(MACD_line)",
            "Histogram   = MACD_line - Signal_line",
            "",
            "Bullish crossover: MACD_line crosses above Signal_line (histogram turns +ve)",
            "Bearish crossover: MACD_line crosses below Signal_line (histogram turns -ve)",
        ],
        source="Appel, G. 'System and Forecasts' newsletter (1979); 'Power Tools for Active Investors' (2005).",
        rationale=(
            "MACD captures the relationship between two different EMA windows to detect "
            "changes in momentum direction, strength, and duration. The histogram — the "
            "difference between the MACD line and its Signal line — is the most sensitive "
            "component: growing positive bars indicate accelerating bullish momentum, while "
            "shrinking bars warn of momentum fading before price itself reverses."
        ),
        analysis=(
            f"MACD Line = {v('MACD_Line')}, Signal = {v('MACD_Signal')}, "
            f"Histogram = {v('MACD_Hist')}. {sig('MACD')}"
        ),
        signal=sig("MACD"),
    )

    # ── 5. Bollinger Bands ────────────────────────────────────────────────────
    story += _indicator_block(
        name="5. Bollinger Bands (20-period, 2 std deviations)",
        current_value=f"Upper={v('BB_Upper')}  Middle={v('BB_Middle')}  Lower={v('BB_Lower')}  %B={v('BB_PctB')}",
        formula_lines=[
            "BB_Middle = SMA_20",
            "BB_Upper  = SMA_20 + (2 x sigma_20)",
            "BB_Lower  = SMA_20 - (2 x sigma_20)",
            "%B        = (Close - BB_Lower) / (BB_Upper - BB_Lower)",
            "Bandwidth = (BB_Upper - BB_Lower) / BB_Middle",
            "",
            "Where:  sigma_20 = rolling 20-day standard deviation of Close",
        ],
        source="Bollinger, J. 'Bollinger on Bollinger Bands' (2001), McGraw-Hill.",
        rationale=(
            "Bollinger Bands adapt to market volatility by expanding when volatility is "
            "high and contracting during quiet periods. The %B indicator places price "
            "within the band: %B above 1.0 means price is above the upper band "
            "(overextended); %B below 0 means price is below the lower band (oversold). "
            "A 'Bollinger Squeeze' — when bandwidth reaches a multi-month low — typically "
            "precedes a large directional breakout, though direction must be determined "
            "from other indicators."
        ),
        analysis=(
            f"Price is at {v('BB_PctB')} of the band range (%B). "
            f"Upper: {v('BB_Upper')}, Middle: {v('BB_Middle')}, Lower: {v('BB_Lower')}. "
            f"{sig('Bollinger')}"
        ),
        signal=sig("Bollinger"),
    )

    # ── 6. Fibonacci Retracement ──────────────────────────────────────────────
    fib_vals = "  |  ".join(f"{k}: ${v_}" for k, v_ in list(fibs.items())[:5]) if fibs else "N/A"
    story += _indicator_block(
        name="6. Fibonacci Retracement Levels",
        current_value=fib_vals,
        formula_lines=[
            "Level = High - (High - Low) x ratio",
            "",
            "Ratios used:  23.6%  (1 - 0.764)  —  shallow pullback",
            "              38.2%  (1 - 0.618)  —  moderate pullback",
            "              50.0%  (0.5)         —  half-way level",
            "              61.8%  (Golden Ratio) —  deep pullback, strongest level",
            "              78.6%  (1 - 0.214)  —  very deep pullback",
            "",
            "Where:  High, Low = recent significant swing high and low",
        ],
        source="Fibonacci, L. 'Liber Abaci' (c. 1202); Prechter & Frost 'Elliott Wave Principle' (1978).",
        rationale=(
            "Fibonacci ratios emerge from the Fibonacci sequence (1, 1, 2, 3, 5, 8, 13 ...) "
            "where each term is the sum of the two preceding. As the sequence grows, the "
            "ratio of consecutive terms approaches 1.618 (the Golden Ratio). In price "
            "action, pullbacks from a trend often find support or resistance at these "
            "levels because many market participants place orders there, creating a "
            "self-fulfilling dynamic. The 61.8% level is considered the strongest."
        ),
        analysis=(
            "Fibonacci levels computed over the full 2-year price history. "
            "Key levels: " + (", ".join(f"{k} = ${v_}" for k, v_ in fibs.items()) if fibs else "data unavailable")
        ),
        signal="See levels table",
    )

    # Fibonacci table
    if fibs:
        fib_rows = [[lbl, f"${price_}"] for lbl, price_ in fibs.items()]
        cw = [USABLE_W * 0.4, USABLE_W * 0.6]
        story.append(_generic_table(["Fib Level", "Price"], fib_rows, cw))
        story.append(_sp(0.3))

    # ── 7. Support & Resistance ───────────────────────────────────────────────
    res_lvls = pivots.get("resistance_levels", [])
    sup_lvls = pivots.get("support_levels",    [])
    res_str  = ", ".join(f"${x}" for x in res_lvls) or "N/A"
    sup_str  = ", ".join(f"${x}" for x in sup_lvls) or "N/A"
    story += _indicator_block(
        name="7. Local Support & Resistance Pivots",
        current_value=f"Resistance: {res_str}  |  Support: {sup_str}",
        formula_lines=[
            "Swing HIGH at index i:  High[i] = max(High[i-w : i+w+1])",
            "Swing LOW  at index i:  Low[i]  = min(Low[i-w  : i+w+1])",
            "",
            "Where:  w = look-back/look-forward window = 10 bars",
            "De-duplication: merge levels within 0.5% of each other",
        ],
        source="Williams, L. 'Long-Term Secrets to Short-Term Trading' (2nd ed., 2011).",
        rationale=(
            "Swing high and low pivots mark price levels at which the market previously "
            "reversed direction. At a prior high, sellers had previously outpaced buyers — "
            "that supply is likely to re-emerge if price returns to that level (resistance). "
            "Conversely, prior lows represent demand zones where buyers previously stepped in "
            "(support). These levels are actionable: traders set entries/exits around them "
            "and risk managers use them to define stop-loss placements."
        ),
        analysis=(
            f"Resistance zones: {res_str}. "
            f"Support zones: {sup_str}. "
            "Price approaching a resistance level from below requires increased caution; "
            "price holding above a support level confirms strength."
        ),
        signal="See levels above",
    )

    return story


def _build_fundamental_section(fundamental: dict, info: dict) -> list:
    """
    Fundamental Analysis section: 5 metrics each with formula, derivation,
    and current stock-specific reading.
    """
    pe     = fundamental.get("pe",     {})
    peg    = fundamental.get("peg",    {})
    graham = fundamental.get("graham", {})
    dcf    = fundamental.get("dcf",    {})
    pb_div = fundamental.get("pb_div", {})

    price = info.get("currentPrice") or 0

    story = [_section_bar("FUNDAMENTAL ANALYSIS"), _sp(0.3)]
    intro = (
        "Fundamental analysis values a company based on its financial statements, "
        "earnings, and cash flows. Unlike technical analysis (which looks at price "
        "action), fundamental metrics ask: 'Is the business itself worth the price "
        "the market is charging?' All five metrics below draw from the Yahoo Finance "
        "company info fields (balance sheet, income statement, cash flow statement)."
    )
    story.append(_p(intro, "body"))
    story.append(_sp(0.2))

    # ── 8. P/E ────────────────────────────────────────────────────────────────
    story += _indicator_block(
        name="8. P/E Ratio — Price to Earnings Multiple",
        current_value=f"Trailing={_fmt(pe.get('trailing_pe'))}x  Forward={_fmt(pe.get('forward_pe'))}x",
        formula_lines=[
            "Trailing P/E = Current Price / Trailing EPS  (last 12 months reported EPS)",
            "Forward  P/E = Current Price / Forward EPS   (next 12 months consensus EPS est.)",
            "",
            "Where:  EPS = Earnings Per Share = Net Income / Diluted Shares Outstanding",
        ],
        source="Graham, B. & Dodd, D. 'Security Analysis' (1934), Ch. 26; Damodaran, A. 'Investment Valuation' (2012).",
        rationale=(
            "The P/E ratio tells you how many dollars of price you pay for each dollar of "
            "annual earnings. It is the most widely-used valuation multiple in equity markets. "
            "A high P/E can mean: (a) the market expects strong future growth, (b) the stock "
            "is overvalued, or both. Thresholds are sector-dependent: tech companies typically "
            "trade at 25-40x while utilities trade at 12-18x. Comparing to the stock's own "
            "5-year average P/E and sector median P/E gives the most meaningful context."
        ),
        analysis=(
            f"Trailing P/E = {_fmt(pe.get('trailing_pe'))}x — {pe.get('trailing_signal','N/A')}. "
            f"Forward P/E = {_fmt(pe.get('forward_pe'))}x — {pe.get('forward_signal','N/A')}."
        ),
        signal=pe.get("trailing_signal", "N/A"),
    )

    # ── 9. PEG ────────────────────────────────────────────────────────────────
    story += _indicator_block(
        name="9. PEG Ratio — Price / Earnings-to-Growth",
        current_value=_fmt(peg.get("peg")),
        formula_lines=[
            "PEG = (P/E ratio) / (Annual EPS Growth Rate x 100)",
            "",
            "Where:  EPS Growth Rate is sourced from Yahoo Finance 'earningsGrowth'",
            "        (year-over-year EPS growth, expressed as a decimal e.g. 0.15 = 15%)",
            "",
            "Rule of thumb:  PEG < 1.0  →  undervalued relative to growth",
            "                PEG = 1.0  →  fairly valued",
            "                PEG > 2.0  →  expensive relative to growth",
        ],
        source="Lynch, P. 'One Up on Wall Street' (1989), pp. 198-199; Damodaran, A. 'Investment Valuation' (2012), Ch. 18.",
        rationale=(
            "The P/E ratio's biggest weakness is that it penalises high-growth companies — "
            "a company growing earnings at 40% per year deserves a higher P/E than one growing "
            "at 5%. The PEG ratio corrects for this by dividing P/E by the growth rate, placing "
            "high-growth and low-growth companies on a comparable scale. A PEG of 1 means you "
            "are paying 1x the growth rate in P/E, which Lynch considered 'fairly valued.' "
            "Note: PEG is unreliable for companies with negative earnings or negative growth."
        ),
        analysis=(
            f"PEG = {_fmt(peg.get('peg'))}. {peg.get('signal','N/A')}"
        ),
        signal=peg.get("signal", "N/A"),
    )

    # ── 10. Graham Number ─────────────────────────────────────────────────────
    gn = graham.get("graham_number")
    story += _indicator_block(
        name="10. Graham Number — Conservative Intrinsic Value Floor",
        current_value=f"${_fmt(gn, null='N/A')}  (Price: ${price:.2f})" if gn else "N/A",
        formula_lines=[
            "Graham Number = sqrt( 22.5 x EPS x BVPS )",
            "",
            "Where:  EPS  = Trailing Earnings Per Share          (from income statement)",
            "        BVPS = Book Value Per Share                 (from balance sheet)",
            "        22.5 = 15 (Graham's max P/E) x 1.5 (Graham's max P/B)",
            "",
            "Margin of Safety = (Graham Number - Current Price) / Current Price x 100%",
        ],
        source="Graham, B. 'The Intelligent Investor' (1973), Ch. 14 — 'Stock Selection for the Defensive Investor'.",
        rationale=(
            "Benjamin Graham designed this formula as a simple upper-bound check for value "
            "investors. It combines two constraints: the stock should not trade above 15x "
            "earnings AND should not trade above 1.5x book value. Multiplying them gives the "
            "22.5 constant. The square root converts the product of two per-share quantities "
            "back into per-share dollars. A stock trading significantly above its Graham "
            "Number fails Graham's strict tests for the 'defensive investor', suggesting "
            "either a growth premium or overvaluation."
        ),
        analysis=(
            f"Graham Number = ${_fmt(gn, null='N/A')}. {graham.get('signal','N/A')}"
        ),
        signal=graham.get("signal", "N/A"),
    )

    # ── 11. DCF ───────────────────────────────────────────────────────────────
    dcf_val = dcf.get("dcf_value")
    story += _indicator_block(
        name="11. Simplified Discounted Cash Flow (DCF)",
        current_value=f"${_fmt(dcf_val, null='N/A')}  (Price: ${price:.2f})" if dcf_val else "N/A",
        formula_lines=[
            "Step 1 — Project 10 years of FCF:",
            "         FCF_t = FCF_0 x (1 + g)^t    for t = 1 to 10",
            "",
            "Step 2 — Terminal Value (Gordon Growth Model):",
            "         TV = FCF_10 x (1 + g_T) / (WACC - g_T)",
            "",
            "Step 3 — Intrinsic Value Per Share:",
            "         IV = [ Sum(FCF_t / (1+WACC)^t) + TV/(1+WACC)^10 ] / Shares",
            "",
            f"Parameters: g = {dcf.get('growth_rate_used','N/A')} growth, "
            f"WACC = {dcf.get('wacc_used','N/A')}, g_T = 3% terminal growth",
        ],
        source="Damodaran, A. 'Investment Valuation' (2002), Ch. 12; Gordon, M. 'The Investment, Financing, and Valuation of the Corporation' (1962).",
        rationale=(
            "The DCF is theoretically the correct intrinsic valuation method: a business is "
            "worth the present value of all the cash it will generate for its owners in the "
            "future. The WACC (Weighted Average Cost of Capital) acts as the discount rate, "
            "representing the opportunity cost of the capital invested. The terminal value, "
            "computed via the Gordon Growth Model, captures the value beyond year 10 and "
            "typically accounts for 60-80% of the total intrinsic value, making the terminal "
            "growth rate assumption critical. A higher WACC = lower intrinsic value."
        ),
        analysis=(
            f"DCF intrinsic value = ${_fmt(dcf_val, null='N/A')}. "
            f"Growth rate used: {dcf.get('growth_rate_used','N/A')}, "
            f"WACC: {dcf.get('wacc_used','N/A')}. "
            f"{dcf.get('signal','N/A')}"
        ),
        signal=dcf.get("signal", "N/A"),
    )

    # ── 12. P/B & Dividend Yield ──────────────────────────────────────────────
    dy = pb_div.get("dividend_yield")
    story += _indicator_block(
        name="12. Price-to-Book (P/B) & Dividend Yield",
        current_value=(
            f"P/B={_fmt(pb_div.get('price_to_book'))}x  "
            f"Div Yield={_fmt(pb_div.get('dividend_yield'), suffix='%', null='N/A')}"
        ),
        formula_lines=[
            "P/B Ratio    = Current Price / Book Value Per Share",
            "             = Market Cap / Total Shareholders' Equity",
            "",
            "Dividend Yield = Annual Dividend Per Share / Current Price  x 100%",
            "",
            "Where:  Book Value Per Share = (Total Assets - Total Liabilities) / Shares",
        ],
        source="Graham (P/B); Siegel, J. 'Stocks for the Long Run' (2014), Ch. 6 (Dividend Yield).",
        rationale=(
            "P/B below 1.0 means the market values the firm below its net accounting assets — "
            "this can represent deep value OR financial distress (seek context). Asset-light "
            "companies (software, pharma) naturally trade at high P/B because their economic "
            "value lies in intangibles not on the balance sheet. "
            "Dividend Yield is the cash return you earn while holding the stock, making it "
            "directly comparable to a savings account or bond. A yield above the risk-free "
            "rate (5.25%) provides income compensation for equity risk; a yield near zero "
            "means the investment case rests entirely on price appreciation."
        ),
        analysis=(
            f"P/B = {_fmt(pb_div.get('price_to_book'))}x — {pb_div.get('pb_signal','N/A')}. "
            f"Dividend Yield = {_fmt(pb_div.get('dividend_yield'), suffix='%', null='N/A')} — "
            f"{pb_div.get('dividend_signal','N/A')}."
        ),
        signal=pb_div.get("pb_signal", "N/A"),
    )

    return story


def _build_statistical_section(statistical: dict) -> list:
    """
    Statistical Analysis section: 5 models each with derivation,
    plus prediction tables for linear regression and Monte Carlo.
    """
    vol  = statistical.get("volatility",  {})
    beta = statistical.get("beta",        {})
    sha  = statistical.get("sharpe",      {})
    reg  = statistical.get("regression",  {})
    mc   = statistical.get("monte_carlo", {})

    story = [_section_bar("STATISTICAL & QUANTITATIVE ANALYSIS"), _sp(0.3)]
    intro = (
        "Statistical models treat price returns as random variables and apply probability "
        "theory to quantify risk, relative performance, trend strength, and future price "
        "distributions. The five models below form a complete risk-return assessment: "
        "volatility measures absolute risk, beta measures relative market risk, Sharpe Ratio "
        "measures risk-adjusted return quality, linear regression extrapolates the trend, "
        "and Monte Carlo GBM simulates thousands of possible futures."
    )
    story.append(_p(intro, "body"))
    story.append(_sp(0.2))

    sigma_a = vol.get("sigma_annual")
    sigma_pct = f"{sigma_a*100:.2f}%" if isinstance(sigma_a, float) else "N/A"

    # ── 13. Historical Volatility ─────────────────────────────────────────────
    story += _indicator_block(
        name="13. Historical (Realised) Volatility",
        current_value=f"sigma_daily={_fmt(vol.get('sigma_daily'), null='N/A')}  sigma_annual={sigma_pct}",
        formula_lines=[
            "Daily log-return:   r_t = ln( Close_t / Close_(t-1) )",
            "Daily volatility:   sigma_daily  = std(r_t)              (sample std dev)",
            "Annual volatility:  sigma_annual = sigma_daily x sqrt(252)",
            "",
            "Where:  252 = approximate number of trading days per year",
            "        std = sample standard deviation of the log-return series",
        ],
        source="Hull, J. 'Options, Futures, and Other Derivatives' (10th ed., 2018), Ch. 15.",
        rationale=(
            "Volatility is the fundamental measure of investment risk in modern finance. "
            "Multiplying daily sigma by sqrt(252) annualises it using the square-root-of-time "
            "rule, which holds when daily returns are independently distributed. "
            "Log-returns are used instead of simple returns because they are time-additive — "
            "the 1-year log-return equals the sum of 252 daily log-returns. "
            "This sigma feeds directly into both the Sharpe Ratio and the GBM simulation."
        ),
        analysis=(
            f"Annualised volatility = {sigma_pct}. {vol.get('label','N/A')}. "
            "High volatility implies wider confidence intervals in the Monte Carlo forecasts "
            "and is penalised in the Sharpe Ratio."
        ),
        signal=vol.get("label", "N/A"),
    )

    # ── 14. Beta ─────────────────────────────────────────────────────────────
    story += _indicator_block(
        name="14. Beta — Systematic Market Risk (CAPM)",
        current_value=_fmt(beta.get("beta")),
        formula_lines=[
            "Beta = Cov(R_stock, R_market) / Var(R_market)",
            "",
            "Where:  R_stock  = daily log-returns of the stock",
            "        R_market = daily log-returns of SPY (S&P 500 ETF)",
            "        Cov()    = sample covariance",
            "        Var()    = sample variance",
            "        Period: 1 year of overlapping trading days",
        ],
        source="Sharpe, W.F. 'Capital Asset Prices: A Theory of Market Equilibrium' (1964); Lintner, J. (1965). CAPM framework.",
        rationale=(
            "Beta decomposes a stock's total risk into two components: systematic risk "
            "(market-wide — unavoidable) and idiosyncratic risk (company-specific — "
            "diversifiable). CAPM states that only systematic risk deserves compensation "
            "via expected return. Beta = 1 means the stock moves dollar-for-dollar with "
            "the S&P 500. Beta > 1 amplifies both gains and losses; Beta < 1 dampens them. "
            "Beta < 0 (rare) means the stock tends to rise when the market falls — "
            "this is a natural portfolio hedge."
        ),
        analysis=(
            f"Beta vs S&P 500 = {_fmt(beta.get('beta'))}. {beta.get('label','N/A')}. "
            "In a market rally, this stock is expected to "
            + (f"outperform by a factor of {_fmt(beta.get('beta'))}x."
               if (beta.get("beta") or 0) > 1
               else "underperform, offering relative downside protection.")
        ),
        signal=beta.get("label", "N/A"),
    )

    # ── 15. Sharpe Ratio ──────────────────────────────────────────────────────
    story += _indicator_block(
        name="15. Sharpe Ratio — Risk-Adjusted Return",
        current_value=_fmt(sha.get("sharpe")),
        formula_lines=[
            "Sharpe = (R_annual - R_f) / sigma_annual",
            "",
            "Where:  R_annual  = mean(r_t) x 252    (annualised historical return)",
            "        R_f       = 0.0525              (5.25% US T-Bill — risk-free rate)",
            "        sigma_annual = sigma_daily x sqrt(252)",
            "",
            "Interpretation:  > 2.0 = Excellent  |  1.0-2.0 = Good",
            "                 0-1.0 = Modest      |  < 0     = Underperforms risk-free",
        ],
        source="Sharpe, W.F. 'Mutual Fund Performance' (1966); 'The Sharpe Ratio', J. Portfolio Management (1994).",
        rationale=(
            "The Sharpe Ratio answers: 'How much return am I getting per unit of risk taken?' "
            "Subtracting R_f normalises out the time-value of money — you must earn more "
            "than the risk-free rate to justify owning the stock at all. Dividing by sigma "
            "penalises volatile assets: two stocks with the same raw return but different "
            "volatilities get different Sharpe Ratios. This is the correct way to compare "
            "assets across risk levels, and it underpins the Capital Market Line in CAPM."
        ),
        analysis=(
            f"Sharpe Ratio = {_fmt(sha.get('sharpe'))}. "
            f"Historical annual return = {sha.get('annual_return','N/A')} vs "
            f"risk-free rate of 5.25%. {sha.get('signal','N/A')}"
        ),
        signal=sha.get("signal", "N/A"),
    )

    # ── 16. Linear Regression ─────────────────────────────────────────────────
    reg_preds = reg.get("predictions", {})
    story += _indicator_block(
        name="16. Linear Regression on Log-Prices (Trend Extrapolation)",
        current_value=f"R-squared={_fmt(reg.get('r_squared'))}  Trend={reg.get('trend_dir','N/A')}",
        formula_lines=[
            "Model:  ln(P_t) = alpha + beta_trend x t + epsilon",
            "Fit via: Ordinary Least Squares (OLS) — minimise sum of squared residuals",
            "",
            "Predicted price:  P(T+h) = exp( alpha_hat + beta_hat x (T + h) )",
            "",
            "Where:  t       = integer day index  (0, 1, 2, ... T)",
            "        alpha   = intercept  (estimated by OLS)",
            "        beta_trend = slope  (estimated by OLS)",
            "        h       = forecast horizon in trading days",
        ],
        source="Gauss, C.F. 'Theoria Motus' (1809, OLS); Lo & MacKinlay 'A Non-Random Walk Down Wall Street' (1999).",
        rationale=(
            "Fitting a straight line to log-prices models prices as growing (or declining) "
            "at a constant exponential rate — the beta_trend coefficient IS the daily log "
            "growth rate. Log-transforming prices first linearises the compounding, so a "
            "straight line in log-space corresponds to exponential growth in price-space. "
            "R² measures how well the historical data lies on this trend: R² near 1.0 means "
            "the stock has trended consistently; R² near 0 means it has been erratic."
        ),
        analysis=(
            f"The R² of {_fmt(reg.get('r_squared'))} indicates a "
            + ("strong" if (reg.get("r_squared") or 0) > 0.7 else
               "moderate" if (reg.get("r_squared") or 0) > 0.4 else "weak")
            + " linear trend fit. Trend direction: "
            + f"{reg.get('trend_dir','N/A')}. "
            "Note: this is a deterministic extrapolation — it does not model uncertainty."
        ),
        signal=f"{reg.get('trend_dir','N/A')} trend (R²={_fmt(reg.get('r_squared'))})",
    )

    if reg_preds:
        story.append(_p("<b>Linear Regression Price Predictions</b>", "h3"))
        rows = []
        for h_label, data in reg_preds.items():
            sign = "+" if data["change%"] >= 0 else ""
            rows.append([h_label, f"${data['price']}", f"{sign}{data['change%']}%"])
        cw = [USABLE_W / 3] * 3
        story.append(_generic_table(["Horizon", "Target Price", "Expected Change"], rows, cw))
        story.append(_sp(0.25))

    # ── 17. Monte Carlo GBM ───────────────────────────────────────────────────
    mc_preds = mc.get("predictions", {})
    mc_params = mc.get("params", {})
    story += _indicator_block(
        name="17. Monte Carlo Simulation — Geometric Brownian Motion (GBM)",
        current_value=(
            f"mu_annual={_fmt(mc_params.get('mu_annual'), suffix='')}"
            f"  sigma_annual={_fmt(mc_params.get('sigma_annual'))}"
            f"  n_paths={mc_params.get('n_paths', 1000):,}"
        ),
        formula_lines=[
            "GBM exact discrete solution:",
            "S(t+1) = S(t) x exp[ (mu - sigma^2/2) x dt + sigma x sqrt(dt) x Z ]",
            "",
            "Where:  mu    = mean(daily log-returns) x 252  (annualised drift from history)",
            "        sigma = std(daily log-returns) x sqrt(252) (annualised volatility)",
            "        dt    = 1/252  (one trading day timestep)",
            "        Z     ~ N(0,1)  (standard normal random draw, new each day)",
            "        S(0)  = current market price",
            "",
            "Run 1,000 independent paths; report P10 (bear), P50 (median), P90 (bull)",
        ],
        source="Black & Scholes (1973); Samuelson, P. 'Rational Theory of Warrant Pricing' (1965); Glasserman, P. 'Monte Carlo Methods in Financial Engineering' (2004), Ch. 3.",
        rationale=(
            "GBM is the standard continuous-time stochastic process for equity prices. "
            "The term (mu - sigma^2/2) is the Ito drift correction, which ensures GBM "
            "reproduces the correct expected price path in continuous time. The sigma*sqrt(dt)*Z "
            "term is the stochastic (random) component — every day a new random shock is "
            "applied. Running 1,000 independent paths creates a distribution of possible "
            "futures rather than a single point forecast — the P10 conveys a realistic bear "
            "scenario, the P90 a realistic bull scenario, and their spread quantifies "
            "how much uncertainty compounds over each horizon."
        ),
        analysis=(
            f"Using historical drift of {_fmt(mc_params.get('mu_annual'), suffix='')}"
            f" and volatility of {_fmt(mc_params.get('sigma_annual'))} "
            f"across {mc_params.get('n_paths',1000):,} simulated paths. "
            "Wider P10-P90 spread at longer horizons reflects compounding uncertainty."
        ),
        signal="Probabilistic — see table below",
    )

    if mc_preds:
        story.append(_p("<b>Monte Carlo GBM Price Forecasts (1,000 Paths)</b>", "h3"))
        rows = []
        for h_label, data in mc_preds.items():
            rows.append([
                h_label,
                f"${data['p10']}  ({data['p10_chg%']:+.1f}%)",
                f"${data['median']}  ({data['med_chg%']:+.1f}%)",
                f"${data['p90']}  ({data['p90_chg%']:+.1f}%)",
            ])
        cw = [USABLE_W * 0.22, USABLE_W * 0.26, USABLE_W * 0.26, USABLE_W * 0.26]
        story.append(_generic_table(
            ["Horizon", "Bear (P10)", "Median (P50)", "Bull (P90)"],
            rows, cw,
        ))
        story.append(_sp(0.3))

    return story


def _build_charts_section(dir_path: str) -> list:
    """Embed the three pre-generated PNG charts."""
    story = [_section_bar("CHARTS"), _sp(0.3)]
    chart_defs = [
        ("price_chart.png",       "Price History with Moving Averages (SMA-20/50/200) & Bollinger Bands"),
        ("technical_chart.png",   "Technical Indicators: Price  |  RSI(14)  |  MACD + Signal + Histogram"),
        ("monte_carlo_chart.png", "Monte Carlo GBM Simulation: 1,000 Price Paths with P10 / P50 / P90 Bands"),
    ]
    for filename, label in chart_defs:
        path = os.path.join(dir_path, filename)
        story += _embed_chart(path, label)
        story.append(PageBreak())
    return story


def _build_disclaimer() -> list:
    """Standard disclaimer flowables."""
    text = (
        "DISCLAIMER — This report was generated by FinBot for educational and research "
        "purposes only. It does not constitute financial advice, investment recommendations, "
        "or an offer to buy or sell any security. All models are mathematical simplifications "
        "of complex market dynamics. Past performance is not indicative of future results. "
        "Monte Carlo simulations, linear regression, and all other quantitative models carry "
        "inherent assumptions and limitations. Always conduct your own due diligence and "
        "consult a qualified financial adviser before making investment decisions."
    )
    return [
        _hr(),
        _sp(0.2),
        Paragraph(text, ST["disclaim"]),
        _sp(0.15),
        Paragraph(
            f"Generated by FinBot  ·  {datetime.date.today().isoformat()}  ·  "
            "Data: Yahoo Finance  ·  LLM: Azure OpenAI",
            ST["disclaim"],
        ),
    ]


# ── Master function ───────────────────────────────────────────────────────────

def generate_pdf_report(
    ticker: str,
    info: dict,
    technical: dict,
    fundamental: dict,
    statistical: dict,
    analyst_data: dict,
    llm: dict,
    dir_path: str,
) -> str:
    """
    Build and save the complete PDF report to dir_path/report.pdf.

    Parameters
    ----------
    ticker, info, technical, fundamental, statistical, analyst_data, llm
        Outputs from their respective fetch/compute functions.
    dir_path : str
        Directory where report.pdf (and the chart PNGs) are saved.

    Returns
    -------
    str — absolute path to the saved report.pdf file.
    """
    pdf_path = os.path.join(dir_path, "report.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize      = A4,
        leftMargin    = MARGIN,
        rightMargin   = MARGIN,
        topMargin     = MARGIN,
        bottomMargin  = MARGIN,
        title         = f"FinBot Analysis: {ticker} — {datetime.date.today().isoformat()}",
        author        = "FinBot",
        subject       = f"Quantitative stock analysis report for {ticker}",
    )

    story = []

    # ── 1. Cover ──────────────────────────────────────────────────────────────
    story.extend(_build_cover(ticker, info, llm))
    story.append(PageBreak())

    # ── 2. Executive Summary ──────────────────────────────────────────────────
    story.extend(_build_executive_summary(llm, analyst_data, info))
    story.append(PageBreak())

    # ── 3. Technical Analysis ─────────────────────────────────────────────────
    story.extend(_build_technical_section(technical))
    story.append(PageBreak())

    # ── 4. Fundamental Analysis ───────────────────────────────────────────────
    story.extend(_build_fundamental_section(fundamental, info))
    story.append(PageBreak())

    # ── 5. Statistical Analysis ───────────────────────────────────────────────
    story.extend(_build_statistical_section(statistical))
    story.append(PageBreak())

    # ── 6. Charts ─────────────────────────────────────────────────────────────
    story.extend(_build_charts_section(dir_path))

    # ── 7. Disclaimer ─────────────────────────────────────────────────────────
    story.extend(_build_disclaimer())

    doc.build(story)
    return pdf_path
