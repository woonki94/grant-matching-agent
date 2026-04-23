from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, List


def build_pdf_filename(subject: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", str(subject or "").strip())
    base = base.strip("._-")
    if not base:
        base = "grant_justification"
    return f"{base[:96]}.pdf"


def build_styled_text_pdf_bytes(text: str) -> bytes:
    """
    Build a lightweight styled PDF (markdown-like) without external dependencies.
    Intended for attaching justification content in a readable format.
    """
    page_w = 612.0
    page_h = 792.0
    margin_x = 46.0
    margin_top = 52.0
    margin_bottom = 46.0

    src = str(text or "")
    src = src.replace("\r\n", "\n").replace("\r", "\n")
    if "\n" not in src and "\\n" in src:
        # Handle payloads where newline escapes arrive as literal characters.
        src = src.replace("\\n", "\n")
    src = src.replace("\t", "    ")

    def _normalize_pdf_text(value: str) -> str:
        out = str(value or "")
        out = out.replace("\u2013", "-").replace("\u2014", "-")
        out = out.replace("\u2018", "'").replace("\u2019", "'")
        out = out.replace("\u201c", '"').replace("\u201d", '"')
        out = out.replace("\u2022", "-")
        # Resolve markdown emphasis markers before PDF rendering.
        out = re.sub(r"\*\*(.+?)\*\*", r"\1", out, flags=re.DOTALL)
        out = out.replace("\\*\\*", "")
        out = out.replace("**", "")
        return out

    def _escape_pdf_line(value: str) -> str:
        clean = _normalize_pdf_text(value).encode("latin-1", "replace").decode("latin-1")
        clean = clean.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        return clean

    styles: Dict[str, Dict[str, Any]] = {
        "title": {
            "font": "F2",
            "size": 22.0,
            "leading": 30.0,
            "color": (0.11, 0.23, 0.55),
            "indent": 0.0,
            "wrap": 50,
        },
        "section": {
            "font": "F2",
            "size": 16.0,
            "leading": 22.0,
            "color": (0.13, 0.25, 0.55),
            "indent": 0.0,
            "wrap": 68,
        },
        "body": {
            "font": "F1",
            "size": 11.5,
            "leading": 17.0,
            "color": (0.16, 0.18, 0.23),
            "indent": 0.0,
            "wrap": 96,
        },
        "bullet": {
            "font": "F1",
            "size": 11.5,
            "leading": 17.0,
            "color": (0.16, 0.18, 0.23),
            "indent": 10.0,
            "wrap": 88,
        },
        "quote": {
            "font": "F3",
            "size": 11.5,
            "leading": 18.0,
            "color": (0.25, 0.31, 0.43),
            "indent": 2.0,
            "wrap": 94,
        },
    }

    tokens: List[Dict[str, Any]] = []
    for raw in src.split("\n"):
        stripped = str(raw or "").strip()
        if not stripped:
            tokens.append({"kind": "spacer", "height": 10.0})
            continue
        if stripped.startswith("# "):
            tokens.append({"kind": "title", "text": stripped[2:].strip()})
            continue
        if stripped.startswith("## "):
            tokens.append({"kind": "section", "text": stripped[3:].strip()})
            continue
        m_bullet = re.match(r"^\s*[-*•]\s+(.+)$", stripped)
        if m_bullet:
            tokens.append({"kind": "bullet", "text": m_bullet.group(1).strip()})
            continue
        tokens.append({"kind": "body", "text": stripped})

    pages: List[List[Dict[str, Any]]] = [[]]
    y = page_h - margin_top

    def _new_page() -> None:
        nonlocal y
        pages.append([])
        y = page_h - margin_top

    def _style_for(kind: str, text_value: str) -> Dict[str, Any]:
        base = dict(styles.get(kind, styles["body"]))
        low = str(text_value or "").lower()
        if kind == "section":
            if "why it fits" in low:
                base["color"] = (0.06, 0.48, 0.22)
            elif "gaps to address" in low:
                base["color"] = (0.80, 0.41, 0.06)
            elif any(x in low for x in ("quick summary", "grant explanation", "full details", "faculty fit")):
                base["color"] = (0.12, 0.26, 0.58)
        if kind == "bullet" and "matching score" in low:
            base["font"] = "F2"
            base["color"] = (0.07, 0.48, 0.23)
        if kind == "body" and ("expertise centers on" in low or "only tangentially touches" in low):
            base = dict(styles["quote"])
        return base

    def _wrap_text(value: str, width: int) -> List[str]:
        normalized = " ".join(_normalize_pdf_text(value).split()).strip()
        if not normalized:
            return [""]
        wrapped = textwrap.wrap(
            normalized,
            width=max(24, int(width or 80)),
            replace_whitespace=True,
            drop_whitespace=True,
            break_long_words=False,
            break_on_hyphens=False,
        )
        return wrapped or [normalized]

    for tok in tokens:
        kind = str(tok.get("kind") or "body")
        if kind == "spacer":
            y -= float(tok.get("height") or 8.0)
            if y < margin_bottom:
                _new_page()
            continue

        text_value = str(tok.get("text") or "").strip()
        if not text_value:
            continue
        style = _style_for(kind, text_value)
        wrapped = _wrap_text(text_value, int(style.get("wrap") or 80))

        for i, chunk in enumerate(wrapped):
            line = chunk
            x = margin_x + float(style.get("indent") or 0.0)
            if kind == "bullet":
                if i == 0:
                    line = f"- {chunk}"
                else:
                    line = f"  {chunk}"
                    x += 6.0
            if y < (margin_bottom + float(style.get("leading") or 16.0)):
                _new_page()
            pages[-1].append(
                {
                    "x": x,
                    "y": y,
                    "text": line,
                    "font": str(style.get("font") or "F1"),
                    "size": float(style.get("size") or 11.0),
                    "leading": float(style.get("leading") or 16.0),
                    "color": tuple(style.get("color") or (0.0, 0.0, 0.0)),
                }
            )
            y -= float(style.get("leading") or 16.0)

        if kind in {"title", "section"}:
            y -= 3.0

    # Object layout:
    # 1 catalog, 2 pages, 3 normal font, 4 bold font, 5 italic font, then [page_obj, content_obj] * N
    objects: List[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")

    page_obj_nums: List[int] = []
    content_obj_nums: List[int] = []
    next_obj = 6
    for _ in pages:
        page_obj_nums.append(next_obj)
        content_obj_nums.append(next_obj + 1)
        next_obj += 2

    kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
    objects.append(f"<< /Type /Pages /Count {len(pages)} /Kids [ {kids} ] >>".encode("latin-1"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >>")

    for idx, lines in enumerate(pages):
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.0f} {page_h:.0f}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> "
            f"/Contents {content_obj_nums[idx]} 0 R >>"
        )
        objects.append(page_obj.encode("latin-1"))

        card_x = margin_x - 14.0
        card_y = margin_bottom - 8.0
        card_w = page_w - (2 * card_x)
        card_h = page_h - margin_top - margin_bottom + 16.0
        ops: List[str] = [
            "q",
            "0.985 0.988 0.997 rg",
            f"0 0 {page_w:.2f} {page_h:.2f} re f",
            "Q",
            "q",
            "1 1 1 rg",
            f"{card_x:.2f} {card_y:.2f} {card_w:.2f} {card_h:.2f} re f",
            "Q",
            "q",
            "0.88 0.93 0.99 rg",
            f"0 {page_h - 38.0:.2f} {page_w:.2f} 38 re f",
            "Q",
            "BT",
        ]

        for row in lines:
            r, g, b = row["color"]
            ops.append(f"/{row['font']} {row['size']:.2f} Tf")
            ops.append(f"{float(r):.3f} {float(g):.3f} {float(b):.3f} rg")
            ops.append(f"1 0 0 1 {float(row['x']):.2f} {float(row['y']):.2f} Tm")
            ops.append(f"({_escape_pdf_line(str(row['text'] or ''))}) Tj")

        ops.extend(
            [
                "ET",
                "BT",
                "/F1 9 Tf",
                "0.44 0.48 0.56 rg",
                f"1 0 0 1 {margin_x:.2f} 24 Tm",
                "(Generated by GrantFetcher) Tj",
                "ET",
            ]
        )

        stream = "\n".join(ops).encode("latin-1", "replace")
        content_obj = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream"
        )
        objects.append(content_obj)

    pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets: List[int] = [0]

    for obj_num, obj_body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{obj_num} 0 obj\n".encode("latin-1")
        pdf += obj_body
        pdf += b"\nendobj\n"

    xref_start = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    pdf += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        pdf += f"{off:010d} 00000 n \n".encode("latin-1")
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    ).encode("latin-1")
    return pdf
