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
    r"[\U0001F300-\U0001F5FF]|[\U0001F600-\U0001F64F]|[\U0001F680-\U0001F6FF]|[\U0001F900-\U0001FAFF]|[\U0001F1E6-\U0001F1FF]|[\u2600-\u26FF]|[\u2700-\u27BF]|[\u2B50-\u2B55]"
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
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            processed_lines.append(line)
            continue

        if in_code_block:
            processed_lines.append(line)
            continue

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
        is_equation_or_code = any(c in stripped for c in ("=", "<", ">", "{", "}", ";", "\\", "$", "^"))
        if (
            len(words) <= 7
            and not is_equation_or_code
            and not stripped.endswith((".", ":", ",", ";", "?", "!"))
            and not stripped.startswith(("-", "*", ">", "$", "1.", "2.", "3.", "#"))
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
    in_code_block = False

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            final_lines.append(line)
            index += 1
            continue

        if in_code_block:
            final_lines.append(line)
            index += 1
            continue

        if len(stripped) == 0:
            final_lines.append("")
            index += 1
            continue

        if VARIABLE_EQUATION_PATTERN.match(stripped):
            variable_group = [stripped]
            next_index = index + 1
            while next_index < len(lines):
                candidate = lines[next_index].strip()
                if candidate.startswith("```") or not VARIABLE_EQUATION_PATTERN.match(candidate):
                    break
                variable_group.append(candidate)
                next_index += 1
            if len(variable_group) >= 2:
                for variable_item in variable_group:
                    final_lines.append(f"- {variable_item}")
                index = next_index
                continue

        if PARAMETER_COLON_PATTERN.match(stripped):
            previous_line = final_lines[-1].strip() if len(final_lines) > 0 else ""
            if previous_line.endswith(":") or previous_line.startswith("- "):
                final_lines.append(f"- {stripped}")
                index += 1
                continue

        final_lines.append(line)
        index += 1

    return "\n".join(final_lines)


def strip_scratchpad_and_thoughts(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thought>.*?</thought>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    lines = cleaned.split("\n")
    first_heading_index = -1
    for idx, line in enumerate(lines):
        if line.strip().startswith("#"):
            first_heading_index = idx
            break

    scratchpad_markers = (
        "drafting section",
        "refining section",
        "sub-topic",
        "heading 1:",
        "heading 2:",
        "heading 3:",
        "markdown headings?",
        "no roman numerals?",
        "no emojis?",
        "academic english?",
        "ignore the",
        "expert academic note taker",
        "irrelevant content:"
    )

    if first_heading_index > 0:
        preamble = "\n".join(lines[:first_heading_index]).lower()
        if any(marker in preamble for marker in scratchpad_markers):
            lines = lines[first_heading_index:]

    checklist_pattern = re.compile(r"^[\w\s\-().]+\?\s*(?:yes|no|checked|true|false)\.?$", re.IGNORECASE)
    drafting_line_pattern = re.compile(r"^(?:Drafting|Refining|Sub-topic|Heading)\s+\d+:", re.IGNORECASE)

    filtered_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if checklist_pattern.match(stripped):
            continue
        if drafting_line_pattern.match(stripped):
            continue
        if stripped.lower().startswith("expert academic note taker."):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def format_clean_markdown(text: str) -> str:
    cleaned = strip_scratchpad_and_thoughts(text)
    cleaned = clean_mojibake_and_emojis(cleaned)
    cleaned = normalize_math_blocks(cleaned)
    cleaned = normalize_headings(cleaned)
    cleaned = normalize_lists_and_definitions(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"
