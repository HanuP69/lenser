import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

CODE_MAX_LINE_LENGTH: int = 88
BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")
MATH_INLINE_PATTERN = re.compile(r"\$([^$\n]+?)\$")
EMOJI_REPLACEMENTS: dict[str, str] = {
    "\u26a0\ufe0f": "[!]",
    "\u26a0": "[!]",
}


LATEX_SYMBOLS: dict[str, str] = {
    "\\times": "\u00d7", "\\cdot": "\u00b7", "\\pm": "\u00b1", "\\div": "\u00f7",
    "\\leq": "<=", "\\le": "<=", "\\geq": ">=", "\\ge": ">=", "\\neq": "!=", "\\ne": "!=",
    "\\approx": "~", "\\sim": "~", "\\to": "->", "\\rightarrow": "->", "\\leftarrow": "<-",
    "\\Rightarrow": "=>", "\\Leftrightarrow": "<=>", "\\infty": "inf", "\\sum": "sum",
    "\\prod": "prod", "\\int": "int", "\\partial": "d", "\\nabla": "grad", "\\in": " in ",
    "\\forall": "for all", "\\exists": "exists", "\\log": "log", "\\ln": "ln", "\\sin": "sin",
    "\\cos": "cos", "\\tan": "tan", "\\max": "max", "\\min": "min", "\\lim": "lim",
    "\\ldots": "...", "\\dots": "...", "\\cdots": "...", "\\mid": "|", "\\land": "AND",
    "\\lor": "OR", "\\neg": "NOT", "\\subset": "subset", "\\cup": "union", "\\cap": "intersect",
}
GREEK_LETTERS: tuple[str, ...] = (
    "alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda", "mu", "pi", "sigma",
    "tau", "phi", "omega", "eta", "rho", "kappa", "nu", "xi", "zeta", "psi", "chi",
    "Gamma", "Delta", "Theta", "Lambda", "Sigma", "Phi", "Omega", "Pi", "Psi"
)
TEXT_COMMAND_PATTERN = re.compile(r"\\(?:text|mathrm|mathbf|mathit|operatorname|mathcal|mathbb)\{([^{}]*)\}")
FRAC_PATTERN = re.compile(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}")
SQRT_PATTERN = re.compile(r"\\sqrt\{([^{}]*)\}")
BRACED_SUPERSCRIPT = re.compile(r"\^\{([^{}]*)\}")
BRACED_SUBSCRIPT = re.compile(r"_\{([^{}]*)\}")
SIMPLE_SUPERSCRIPT = re.compile(r"\^([A-Za-z0-9+\-])")
SIMPLE_SUBSCRIPT = re.compile(r"_([A-Za-z0-9])")


def latex_to_markup(expression: str) -> str:
    text = expression
    for _ in range(4):
        text = TEXT_COMMAND_PATTERN.sub(lambda match: match.group(1), text)
        text = FRAC_PATTERN.sub(lambda match: f"({match.group(1)})/({match.group(2)})", text)
        text = SQRT_PATTERN.sub(lambda match: f"sqrt({match.group(1)})", text)
    text = text.replace("\\left", "").replace("\\right", "")
    for command in sorted(LATEX_SYMBOLS, key=len, reverse=True):
        text = re.sub(re.escape(command) + r"(?![A-Za-z])", lambda match, c=command: LATEX_SYMBOLS[c], text)
    for letter in GREEK_LETTERS:
        text = re.sub(r"\\" + letter + r"(?![A-Za-z])", letter, text)
    text = re.sub(r"\\[,;:!\s]", " ", text)
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    safe = escape(text)
    safe = BRACED_SUPERSCRIPT.sub(lambda match: f"<super>{match.group(1)}</super>", safe)
    safe = BRACED_SUBSCRIPT.sub(lambda match: f"<sub>{match.group(1)}</sub>", safe)
    safe = SIMPLE_SUPERSCRIPT.sub(lambda match: f"<super>{match.group(1)}</super>", safe)
    safe = SIMPLE_SUBSCRIPT.sub(lambda match: f"<sub>{match.group(1)}</sub>", safe)
    return safe.replace("{", "").replace("}", "")


def convert_inline_markup(raw_text: str) -> str:
    replaced = raw_text
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        replaced = replaced.replace(emoji, replacement)
    math_segments: list[str] = []

    def stash_math(match: re.Match) -> str:
        math_segments.append(latex_to_markup(match.group(1)))
        return f"\x00M{len(math_segments) - 1}\x00"

    tokenized = MATH_INLINE_PATTERN.sub(stash_math, replaced)
    safe_text = escape(tokenized)
    bolded = BOLD_PATTERN.sub(lambda match: f"<b>{match.group(1)}</b>", safe_text)
    return re.sub(
        "\x00M(\\d+)\x00",
        lambda match: f"<i>{math_segments[int(match.group(1))]}</i>",
        bolded
    )


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
            story.append(Preformatted(code_content, code_style, maxLineLength=CODE_MAX_LINE_LENGTH))
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
            story.append(Paragraph(latex_to_markup(" ".join(math_lines)), math_style))
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
