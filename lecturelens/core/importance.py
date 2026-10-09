EXAM_RELEVANT_PATTERNS: tuple[str, ...] = (
    "important",
    "remember",
    "key point",
    "key takeaway",
    "exam",
    "test",
    "quiz",
    "midterm",
    "final",
    "make sure you know",
    "pay attention",
    "critical",
    "essential",
    "must know",
    "will be graded",
    "note this down",
    "write this down",
)

DEFINITION_PATTERNS: tuple[str, ...] = (
    "definition",
    "we define",
    "is defined as",
    "means that",
    "refers to",
    "is called",
    "known as",
    "denoted by",
    "the term",
    "formally",
)

EXAMPLE_PATTERNS: tuple[str, ...] = (
    "for example",
    "for instance",
    "let me illustrate",
    "consider the case",
    "such as",
    "suppose we have",
    "let's say",
    "imagine",
    "to demonstrate",
    "as an example",
)

CONFUSION_PATTERNS: tuple[str, ...] = (
    "wait what",
    "i don't understand",
    "can you repeat",
    "confused",
    "what do you mean",
    "sorry what",
    "come again",
    "pardon",
    "could you explain",
    "not clear",
    "i'm lost",
)


def detect_importance_flags(text: str) -> list[str]:
    lower_text = text.lower()
    flags: list[str] = []

    found_exam = False
    for pattern in EXAM_RELEVANT_PATTERNS:
        if pattern in lower_text:
            found_exam = True
            break
    if found_exam:
        flags.append("exam_relevant")

    found_definition = False
    for pattern in DEFINITION_PATTERNS:
        if pattern in lower_text:
            found_definition = True
            break
    if found_definition:
        flags.append("definition")

    found_example = False
    for pattern in EXAMPLE_PATTERNS:
        if pattern in lower_text:
            found_example = True
            break
    if found_example:
        flags.append("example")

    found_confusion = False
    for pattern in CONFUSION_PATTERNS:
        if pattern in lower_text:
            found_confusion = True
            break
    if found_confusion:
        flags.append("confusion")

    return flags
