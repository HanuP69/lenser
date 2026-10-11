import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")
MATH_INLINE_PATTERN = re.compile(r"\$([^$\n]+?)\$")
EMOJI_REPLACEMENTS: dict[str, str] = {
    "\u26a0\ufe0f": "[!]",
    "\u26a0": "[!]",
}


def convert_inline_markup(raw_text: str) -> str:
    replaced = raw_text
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        replaced = replaced.replace(emoji, replacement)
    safe_text = escape(replaced)
    bolded = BOLD_PATTERN.sub(lambda match: f"<b>{match.group(1)}</b>", safe_text)
    return MATH_INLINE_PATTERN.sub(lambda match: f"<i>{match.group(1)}</i>", bolded)


def is_table_separator(row_cells: list[str]) -> bool:
    return len(row_cells) > 0 and all(re.match(r"^:?-+:?$", cell.strip()) is not None for cell in row_cells)


def create_pdf_from_markdown(markdown_text: str, output_file_path: Path, title: str) -> None:
    styles = getSampleStyleSheet()
    quote_style = ParagraphStyle(
        "Quote", parent=styles["BodyText"], leftIndent=18, textColor="#444444"
    )
    bullet_style = ParagraphStyle("Bullet", parent=styles["BodyText"], leftIndent=18, bulletIndent=6)
    code_style = ParagraphStyle(
        "CodeBlock",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        backColor="#f4f4f4",
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=4
    )
    math_style = ParagraphStyle(
        "MathDisplay",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        alignment=1,
        spaceBefore=6,
        spaceAfter=6,
        textColor="#1a237e"
    )
    cell_style = ParagraphStyle("TableCell", parent=styles["BodyText"], fontSize=8.5, leading=11)

    story: list = [Paragraph(convert_inline_markup(title), styles["Title"]), Spacer(1, 0.4 * cm)]
    lines = markdown_text.splitlines()
    line_index = 0

    while line_index < len(lines):
        raw_line = lines[line_index]
        line = raw_line.rstrip()
        stripped = line.strip()

        if len(stripped) == 0:
            story.append(Spacer(1, 0.2 * cm))
            line_index += 1
            continue

        if stripped.startswith("```"):
            code_lines: list[str] = []
            line_index += 1
            while line_index < len(lines) and not lines[line_index].strip().startswith("```"):
                code_lines.append(lines[line_index])
                line_index += 1
            if line_index < len(lines):
                line_index += 1
            code_content = "\n".join(code_lines)
            story.append(Preformatted(code_content, code_style))
            continue

        if stripped.startswith("$$"):
            math_lines: list[str] = []
            if stripped == "$$":
                line_index += 1
                while line_index < len(lines) and lines[line_index].strip() != "$$":
                    math_lines.append(lines[line_index].strip())
                    line_index += 1
                if line_index < len(lines):
                    line_index += 1
            else:
                inline_content = stripped.strip("$").strip()
                math_lines.append(inline_content)
                line_index += 1
            story.append(Paragraph(convert_inline_markup(" ".join(math_lines)), math_style))
            continue

        if stripped.startswith("|"):
            table_raw_rows: list[list[str]] = []
            while line_index < len(lines) and lines[line_index].strip().startswith("|"):
                row_str = lines[line_index].strip()
                cells = [c.strip() for c in row_str.strip("|").split("|")]
                if not is_table_separator(cells):
                    table_raw_rows.append(cells)
                line_index += 1

            if len(table_raw_rows) > 0:
                flowable_table_data = []
                for row in table_raw_rows:
                    flowable_row = [Paragraph(convert_inline_markup(cell), cell_style) for cell in row]
                    flowable_table_data.append(flowable_row)
                reportlab_table = Table(
                    flowable_table_data,
                    style=[
                        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
                story.append(reportlab_table)
                story.append(Spacer(1, 0.3 * cm))
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match is not None:
            level = min(len(heading_match.group(1)), 3)
            style_name = f"Heading{level}"
            story.append(Paragraph(convert_inline_markup(heading_match.group(2)), styles[style_name]))
            line_index += 1
            continue

        if line.lstrip().startswith(">"):
            quoted = line.lstrip()[1:].strip()
            story.append(Paragraph(convert_inline_markup(quoted), quote_style))
            line_index += 1
            continue

        bullet_match = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet_match is not None:
            story.append(Paragraph(convert_inline_markup(bullet_match.group(1)), bullet_style, bulletText="•"))
            line_index += 1
            continue

        story.append(Paragraph(convert_inline_markup(line), styles["BodyText"]))
        line_index += 1

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_file_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title=title
    )
    document.build(story)
