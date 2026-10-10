import re

EXAM_RELEVANT_PATTERNS: tuple[str, ...] = (
    r"important",
    r"remember",
    r"key points?",
    r"key takeaways?",
    r"exams?",
    r"tests?",
    r"quiz(?:zes)?",
    r"midterms?",
    r"finals",
    r"final (?:exam|test|project|paper)",
    r"make sure you know",
    r"pay attention",
    r"critical",
    r"essential",
    r"must know",
    r"will be graded",
    r"note this down",
    r"write this down",
)

DEFINITION_PATTERNS: tuple[str, ...] = (
    r"definitions?",
    r"we define",
    r"is defined as",
    r"means that",
    r"refers to",
    r"is called",
    r"known as",
    r"denoted by",
    r"the term",
    r"formally",
)

EXAMPLE_PATTERNS: tuple[str, ...] = (
    r"for example",
    r"for instance",
    r"let me illustrate",
    r"consider the case",
    r"such as",
    r"suppose we have",
    r"let's say",
    r"imagine",
    r"to demonstrate",
    r"as an example",
)

CONFUSION_PATTERNS: tuple[str, ...] = (
    r"wait,? what",
    r"i don't understand",
    r"can you repeat",
    r"confused",
    r"what do you mean",
    r"sorry,? what",
    r"come again",
    r"pardon",
    r"could you explain",
    r"not clear",
    r"i'm lost",
)


def compile_pattern_group(patterns: tuple[str, ...]) -> re.Pattern:
    alternatives: list[str] = []
    for pattern in patterns:
        alternatives.append(pattern)
    joined = "|".join(alternatives)
    return re.compile(r"(?<![\w'])(?:" + joined + r")(?![\w'])", re.IGNORECASE)


EXAM_RELEVANT_REGEX: re.Pattern = compile_pattern_group(EXAM_RELEVANT_PATTERNS)
DEFINITION_REGEX: re.Pattern = compile_pattern_group(DEFINITION_PATTERNS)
EXAMPLE_REGEX: re.Pattern = compile_pattern_group(EXAMPLE_PATTERNS)
CONFUSION_REGEX: re.Pattern = compile_pattern_group(CONFUSION_PATTERNS)


def detect_importance_flags(text: str) -> list[str]:
    normalized_text = text.replace("\u2019", "'")
    flags: list[str] = []

    if EXAM_RELEVANT_REGEX.search(normalized_text) is not None:
        flags.append("exam_relevant")
    if DEFINITION_REGEX.search(normalized_text) is not None:
        flags.append("definition")
    if EXAMPLE_REGEX.search(normalized_text) is not None:
        flags.append("example")
    if CONFUSION_REGEX.search(normalized_text) is not None:
        flags.append("confusion")

    return flags
