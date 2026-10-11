import re

EXAM_RELEVANT_PATTERNS: tuple[str, ...] = (
    r"important",
    r"remember",
    r"key points?",
    r"key takeaways?",
    r"exams?",
    r"(?:class|mock|written|lab|pop|surprise|chapter)\s+tests?",
    r"(?:on|in|for)\s+(?:the\s+)?tests?\b",
    r"tests?\s+(?:on|for|tomorrow|next\s+(?:week|monday|tuesday|wednesday|thursday|friday)|scheduled|will\s+cover|covers?)",
    r"(?:the\s+)?tests?\s+will\s+cover",
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
    r"महत्वपूर्ण",
    r"परीक्षा",
    r"dh?ya+n (?:do|rakhna)",
    r"exam me",
    r"yaad rakhna",
    r"note kar (?:lo|lena)",
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
    r"परिभाषा",
    r"ka matlab",
    r"bolte hain",
    r"kehte hain",
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
    r"उदाहरण",
    r"jaise ki",
    r"maan lo",
    r"maan lijiye",
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
    r"samajh nahi (?:aaya|aa raha)",
    r"phir se batao",
    r"doubt hai",
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
