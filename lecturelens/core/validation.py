import re

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")
COURSE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\- ]{0,63}$")

VALID_AUDIO_SOURCES: tuple[str, ...] = ("mic", "system", "both")
VALID_LLM_MODES: tuple[str, ...] = ("local", "hybrid", "cloud")
VALID_RUN_MODES: tuple[str, ...] = ("local", "server")
VALID_EXPORT_FORMATS: tuple[str, ...] = ("md",)


def is_valid_session_id(session_id: str) -> bool:
    if SESSION_ID_PATTERN.match(session_id) is None:
        return False
    if ".." in session_id:
        return False
    return True


def is_valid_course_name(course_name: str) -> bool:
    return COURSE_NAME_PATTERN.match(course_name) is not None


def sanitize_file_stem(raw_name: str) -> str:
    cleaned_characters: list[str] = []
    for character in raw_name:
        if character.isalnum() or character in "-_.":
            cleaned_characters.append(character)
        else:
            cleaned_characters.append("_")
    cleaned = "".join(cleaned_characters).strip("._")
    if len(cleaned) == 0:
        return "default"
    return cleaned
