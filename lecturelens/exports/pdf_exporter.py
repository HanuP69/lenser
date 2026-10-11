import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")
EMOJI_REPLACEMENTS: dict[str, str] = {
    "\u26a0\ufe0f": "[!]",
    "\u26a0": "[!]",
}


def convert_inline_markup(raw_text: str) -> str:
    replaced = raw_text
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        replaced = replaced.replace(emoji, replacement)
    safe_text = escape(replaced)
    return BOLD_PATTERN.sub(lambda match: f"<b>{match.group(1)}</b>", safe_text)


def create_pdf_from_markdown(markdown_text: str, output_file_path: Path, title: str) -> None:
    styles = getSampleStyleSheet()
    quote_style = ParagraphStyle(
        "Quote", parent=styles["BodyText"], leftIndent=18, textColor="#444444"
    )
    bullet_style = ParagraphStyle("Bullet", parent=styles["BodyText"], leftIndent=18, bulletIndent=6)

    story: list = [Paragraph(convert_inline_markup(title), styles["Title"]), Spacer(1, 0.4 * cm)]
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if len(line.strip()) == 0:
            story.append(Spacer(1, 0.2 * cm))
            continue
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match is not None:
            level = min(len(heading_match.group(1)), 3)
            style_name = f"Heading{level}"
            story.append(Paragraph(convert_inline_markup(heading_match.group(2)), styles[style_name]))
            continue
        if line.lstrip().startswith(">"):
            quoted = line.lstrip()[1:].strip()
            story.append(Paragraph(convert_inline_markup(quoted), quote_style))
            continue
        bullet_match = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet_match is not None:
            story.append(Paragraph(convert_inline_markup(bullet_match.group(1)), bullet_style, bulletText="•"))
            continue
        story.append(Paragraph(convert_inline_markup(line), styles["BodyText"]))

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_file_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title=title
    )
    document.build(story)
