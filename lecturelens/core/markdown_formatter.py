import re

MOJIBAKE_REPLACEMENTS: dict[str, str] = {
    "ðŸ“": "",
    "ðŸ": "",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€˜": "'",
    "â€™": "'",
    "â€”": " - ",
    "â€“": " - ",
    "â€¦": "...",
    "Ã©": "e",
    "Ã¨": "e",
    "Ã ": "a",
    "Ã§": "c",
}

DISALLOWED_EMOJI_PATTERN: re.Pattern = re.compile(
    r"[\U00010000-\U0010ffff]|[\u2600-\u269f]|[\u26a1-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]"
)

ROMAN_NUMERAL_HEADING_PATTERN: re.Pattern = re.compile(
    r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\.\s+(.+)$"
)

DISPLAY_MATH_PATTERN: re.Pattern = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)

VARIABLE_EQUATION_PATTERN: re.Pattern = re.compile(r"^\$[A-Za-z0-9_]+\s*=\s*[^$]+\$$")

PARAMETER_COLON_PATTERN: re.Pattern = re.compile(r"^(\$[A-Za-z0-9_]+\$|[A-Za-z0-9_]+)\s*:\s*(.+)$")


def clean_mojibake_and_emojis(text: str) -> str:
    cleaned = text
    for bad_sequence, replacement in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad_sequence, replacement)
    return DISALLOWED_EMOJI_PATTERN.sub("", cleaned)


def normalize_math_blocks(text: str) -> str:
    def format_math_match(match: re.Match) -> str:
        equation_content = match.group(1).strip()
        return f"\n\n$$\n{equation_content}\n$$\n\n"

    return DISPLAY_MATH_PATTERN.sub(format_math_match, text)


def normalize_headings(text: str) -> str:
    lines = text.split("\n")
    processed_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if len(stripped) == 0:
            processed_lines.append("")
            continue

        stripped = re.sub(r"\s*\((?:Cont\.?|Continued)\)", "", stripped, flags=re.IGNORECASE)

        if stripped.startswith("#"):
            processed_lines.append(stripped)
            continue

        roman_match = ROMAN_NUMERAL_HEADING_PATTERN.match(stripped)
        if roman_match:
            heading_title = roman_match.group(1).strip()
            processed_lines.append(f"## {heading_title}")
            continue

        words = stripped.split()
        if (
            len(words) <= 7
            and not stripped.endswith((".", ":", ",", ";", "?", "!"))
            and not stripped.startswith(("-", "*", ">", "$", "1.", "2.", "3."))
            and any(w[0].isupper() for w in words if len(w) > 0 and w[0].isalpha())
            and len(stripped) <= 60
        ):
            processed_lines.append(f"## {stripped}")
            continue

        processed_lines.append(stripped)

    return "\n".join(processed_lines)


def normalize_lists_and_definitions(text: str) -> str:
    lines = text.split("\n")
    final_lines: list[str] = []

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if len(line) == 0:
            final_lines.append("")
            index += 1
            continue

        if VARIABLE_EQUATION_PATTERN.match(line):
            variable_group = [line]
            next_index = index + 1
            while next_index < len(lines) and VARIABLE_EQUATION_PATTERN.match(lines[next_index].strip()):
                variable_group.append(lines[next_index].strip())
                next_index += 1
            if len(variable_group) >= 2:
                for variable_item in variable_group:
                    final_lines.append(f"- {variable_item}")
                index = next_index
                continue

        if PARAMETER_COLON_PATTERN.match(line):
            previous_line = final_lines[-1].strip() if len(final_lines) > 0 else ""
            if previous_line.endswith(":") or previous_line.startswith("- "):
                final_lines.append(f"- {line}")
                index += 1
                continue

        final_lines.append(line)
        index += 1

    return "\n".join(final_lines)


def format_clean_markdown(text: str) -> str:
    cleaned = clean_mojibake_and_emojis(text)
    cleaned = normalize_math_blocks(cleaned)
    cleaned = normalize_headings(cleaned)
    cleaned = normalize_lists_and_definitions(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"
