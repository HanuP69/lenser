import re

EXAM_RELEVANT_PATTERNS: tuple[str, ...] = (
    r"(?:very\s+|super\s+|most\s+|really\s+)?important\s+(?:for|in|on)\s+(?:the\s+)?(?:exam|test|quiz|midterm|final|grade|paper)",
    r"(?:this\s+is\s+|note\s+this\s+|mark\s+this\s+as\s+)(?:very\s+)?important",
    r"important\s+(?:to\s+note|takeaway|concept\s+for|question|topic\s+for)",
    r"bahut\s+important",
    r"important\s+hai",
    r"remember\s+this",
    r"remember\s+(?:for\s+(?:the\s+)?exam|well)",
    r"(?:must|always)\s+remember",
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
    r"critical\s+(?:for\s+(?:the\s+)?exam|to\s+remember|concept\s+to\s+know|for\s+your\s+grade)",
    r"essential\s+(?:for\s+(?:the\s+)?exam|to\s+remember|to\s+know\s+for)",
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
    r"(?:term|definition)\s+means\s+that",
    r"by\s+definition\s+means",
    r"specifically\s+means\s+that",
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

AD_PATTERNS: tuple[str, ...] = (
    r"promo\s+codes?",
    r"coupon\s+codes?",
    r"discount\s+codes?",
    r"sponsored\s+by",
    r"brought\s+to\s+you\s+by",
    r"use\s+code\s+[a-z0-9_-]+",
    r"special\s+offers?",
    r"limited\s+time\s+offer",
    r"(?:link|check\s+the\s+link)\s+in\s+the\s+description",
    r"description\s+box\s+below",
    r"(?:get|for)\s+\d+%\s+off",
    r"nordvpn",
    r"expressvpn",
    r"surfshark",
    r"squarespace",
    r"skillshare",
    r"audible(?:\.com)?",
    r"patreon(?:\.com)?",
    r"download\s+(?:the\s+)?app",
    r"subscribe\s+to\s+my\s+channel",
    r"smash\s+that\s+like\s+button",
    r"hit\s+the\s+bell\s+icon",
)

ACTION_ITEM_PATTERNS: tuple[str, ...] = (
    r"homework\s+(?:is|due|assignment)",
    r"assignment\s+(?:is|due|#\d+)",
    r"submit\s+(?:by|before|your)",
    r"due\s+(?:by|before|on|next|tomorrow|friday|monday)",
    r"project\s+submission",
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
AD_REGEX: re.Pattern = compile_pattern_group(AD_PATTERNS)
ACTION_ITEM_REGEX: re.Pattern = compile_pattern_group(ACTION_ITEM_PATTERNS)


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
    if AD_REGEX.search(normalized_text) is not None:
        flags.append("ad")
    if ACTION_ITEM_REGEX.search(normalized_text) is not None:
        flags.append("action_item")

    return flags
