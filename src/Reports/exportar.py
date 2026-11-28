"""Download formats for the medication review report (Person 5).

* ``render_html`` - a self-contained web page. It opens in any browser, needs no
  other application, and has a "Print or save as PDF" button.
* ``render_pdf`` - a PDF built with ReportLab. It opens in Preview, Adobe Reader
  or any browser. The mandatory footer is printed on every page and again at
  the end of the report.

Both read the same ``Report.blocks`` as the on-screen Markdown, so every format
shows the same content.
"""
from __future__ import annotations

import html
import io
from xml.sax.saxutils import escape as xml_escape

from src.Reports.informe import FOOTER, Report

# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       color: #1f2328; background: #f4f5f7; margin: 0; line-height: 1.55; }
main { max-width: 860px; margin: 24px auto; background: #fff; padding: 32px 40px;
       border: 1px solid #e3e5e8; border-radius: 10px; }
h1 { font-size: 26px; margin: 0 0 8px; }
h2 { font-size: 19px; margin: 30px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #e3e5e8; }
h3 { font-size: 15.5px; margin: 18px 0 6px; }
.meta { color: #4b5563; margin: 0 0 16px; }
.banner { background: #fff7e0; border: 1px solid #f0d58a; padding: 10px 14px; border-radius: 8px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 14px; }
th, td { border: 1px solid #e3e5e8; padding: 6px 9px; text-align: left; vertical-align: top; }
th { background: #f3f4f6; font-weight: 600; }
.note { color: #6b7280; font-style: italic; }
blockquote { margin: 6px 0 12px; padding: 6px 14px; border-left: 3px solid #9ca3af;
             background: #f9fafb; color: #374151; }
.footer { margin-top: 28px; padding-top: 12px; border-top: 1px solid #e3e5e8;
          font-style: italic; color: #4b5563; }
.toolbar { text-align: right; margin-bottom: 8px; }
button { font: inherit; font-size: 14px; padding: 6px 14px; border-radius: 6px;
         border: 1px solid #c9ccd1; background: #fff; cursor: pointer; }
@media print {
  body { background: #fff; }
  main { border: none; margin: 0; padding: 0; max-width: none; }
  .toolbar { display: none; }
  h2, h3 { break-after: avoid; }
  table, blockquote { break-inside: avoid; }
}
@page { margin: 18mm; }
"""


def render_html(report: Report) -> str:
    e = html.escape
    parts: list[str] = []
    for b in report.blocks:
        if b.kind == "title":
            parts.append(f"<h1>{e(b.text)}</h1>")
        elif b.kind == "meta":
            parts.append('<p class="meta">' + "<br>".join(
                f"<strong>{e(k)}:</strong> {e(v)}" for k, v in b.items) + "</p>")
        elif b.kind == "banner":
            parts.append(f'<p class="banner"><strong>{e(b.label)}</strong> {e(b.text)}</p>')
        elif b.kind in ("h2", "h3"):
            parts.append(f"<{b.kind}>{e(b.text)}</{b.kind}>")
        elif b.kind == "para":
            parts.append(f"<p>{e(b.text)}</p>")
        elif b.kind == "note":
            parts.append(f'<p class="note">{e(b.text)}</p>')
        elif b.kind == "table":
            head = "".join(f"<th>{e(h)}</th>" for h in b.header)
            body = "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in row) + "</tr>" for row in b.rows)
            parts.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        elif b.kind == "bullets":
            parts.append("<ul>" + "".join(f"<li>{e(i)}</li>" for i in b.items) + "</ul>")
        elif b.kind == "numbered":
            parts.append("<ol>" + "".join(f"<li>{e(i)}</li>" for i in b.items) + "</ol>")
        elif b.kind == "source":
            link = (
                f' ? <a href="{e(b.url)}" target="_blank" rel="noopener noreferrer">'
                'Open CIMA technical sheet</a>'
                if b.url else ""
            )
            parts.append(
                f"<p><strong>{e(b.label)}:</strong> {e(b.text)}{link}</p>"
            )
        elif b.kind == "quote":
            parts.append(f"<blockquote>“{e(b.text)}”</blockquote>")
        elif b.kind == "footer":
            parts.append(f'<p class="footer">{e(b.text)}</p>')
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{e(report.title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n<main>\n"
        '<div class="toolbar"><button onclick="window.print()">Print or save as PDF</button></div>\n'
        + "\n".join(parts)
        + "\n</main>\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

# The built-in PDF fonts cover Western European text (Spanish accents included)
# but not a few symbols. Replace those so nothing prints as a blank box.
_PDF_REPLACEMENTS = {
    "→": "->", "←": "<-", "⚠": "(!)", "≥": ">=", "≤": "<=", "−": "-",
    " ": " ", " ": " ", " ": " ",
}


def _pdf_text(text: str) -> str:
    for old, new in _PDF_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("cp1252", errors="replace").decode("cp1252")


def _p(text: str) -> str:
    """Plain data text -> safe ReportLab paragraph markup."""
    return xml_escape(_pdf_text(" ".join(str(text).split())))


def render_pdf(report: Report) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ink, muted, line = colors.HexColor("#1f2328"), colors.HexColor("#5b6270"), colors.HexColor("#d9dce1")
    base = ParagraphStyle("base", fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=ink)
    st = {
        "title": ParagraphStyle("title", parent=base, fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=6),
        "meta": ParagraphStyle("meta", parent=base, textColor=muted, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=12.5, leading=16,
                             spaceBefore=12, spaceAfter=5, keepWithNext=1),
        "h3": ParagraphStyle("h3", parent=base, fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                             spaceBefore=7, spaceAfter=3, keepWithNext=1),
        "para": ParagraphStyle("para", parent=base, spaceAfter=4),
        "note": ParagraphStyle("note", parent=base, fontName="Helvetica-Oblique", textColor=muted, spaceAfter=4),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8.8, leading=11.5),
        "head": ParagraphStyle("head", parent=base, fontName="Helvetica-Bold", fontSize=8.8, leading=11.5),
        "list": ParagraphStyle("list", parent=base, leftIndent=14, bulletIndent=2, spaceAfter=2),
        "quote": ParagraphStyle("quote", parent=base, fontName="Helvetica-Oblique", textColor=colors.HexColor("#374151")),
        "footer": ParagraphStyle("footer", parent=base, fontName="Helvetica-Oblique", textColor=muted),
    }

    buf = io.BytesIO()
    left = right = 18 * mm
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=left, rightMargin=right, topMargin=20 * mm, bottomMargin=22 * mm,
        title=_pdf_text(report.title), author="Medication Review AI", subject=FOOTER,
    )
    width = A4[0] - left - right

    def boxed(paragraph, *, bg, edge, left_bar=False):
        t = Table([[paragraph]], colWidths=[width])
        style = [("BACKGROUND", (0, 0), (-1, -1), bg),
                 ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                 ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]
        style.append(("LINEBEFORE", (0, 0), (0, -1), 2.5, edge) if left_bar else ("BOX", (0, 0), (-1, -1), 0.6, edge))
        t.setStyle(TableStyle(style))
        return t

    story: list = []
    for b in report.blocks:
        if b.kind == "title":
            story.append(Paragraph(_p(b.text), st["title"]))
        elif b.kind == "meta":
            story.append(Paragraph("<br/>".join(f"<b>{_p(k)}:</b> {_p(v)}" for k, v in b.items), st["meta"]))
        elif b.kind == "banner":
            story += [boxed(Paragraph(f"<b>{_p(b.label)}</b> {_p(b.text)}", st["para"]),
                            bg=colors.HexColor("#fff7e0"), edge=colors.HexColor("#e9c96a")), Spacer(1, 4)]
        elif b.kind in ("h2", "h3", "para", "note"):
            story.append(Paragraph(_p(b.text), st[b.kind]))
        elif b.kind == "table":
            n = len(b.header)
            fractions = {3: [0.55, 0.33, 0.12], 2: [0.78, 0.22]}.get(n, [1 / n] * n)
            data = [[Paragraph(_p(h), st["head"]) for h in b.header]]
            data += [[Paragraph(_p(c), st["cell"]) for c in row] for row in b.rows]
            t = Table(data, colWidths=[f * width for f in fractions], repeatRows=1)
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, line),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story += [t, Spacer(1, 5)]
        elif b.kind == "bullets":
            story += [Paragraph(_p(i), st["list"], bulletText="•") for i in b.items]
        elif b.kind == "numbered":
            story += [Paragraph(_p(i), st["list"], bulletText=f"{n}.") for n, i in enumerate(b.items, 1)]
        elif b.kind == "source":
            link = (
                f' ? <link href="{_p(b.url)}" color="#0f766e">'
                'Open CIMA technical sheet</link>'
                if b.url else ""
            )
            story.append(Paragraph(
                f"<b>{_p(b.label)}:</b> {_p(b.text)}{link}", st["para"]
            ))
        elif b.kind == "quote":
            story += [boxed(Paragraph(f"“{_p(b.text)}”", st["quote"]),
                            bg=colors.HexColor("#f6f7f9"), edge=colors.HexColor("#9ca3af"), left_bar=True),
                      Spacer(1, 6)]
        elif b.kind == "footer":
            rule = Table([[""]], colWidths=[width], rowHeights=[2])
            rule.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.6, line)]))
            story += [Spacer(1, 14), rule, Paragraph(_p(b.text), st["footer"])]

    def on_page(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(muted)
        canvas.drawString(left, A4[1] - 12 * mm, _pdf_text(report.title))
        canvas.drawString(left, 11 * mm, _pdf_text(FOOTER))
        canvas.drawRightString(A4[0] - right, 11 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buf.getvalue()
