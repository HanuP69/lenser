import html
import re
import zlib
from pathlib import Path
from typing import Any

import genanki

MODEL_ID = 1607392319


def compute_stable_deck_id(deck_name: str) -> int:
    checksum = zlib.crc32(deck_name.encode("utf-8"))
    return (1 << 30) + (checksum % (1 << 30))


def convert_markdown_math_to_anki(text: str) -> str:
    display_converted = re.sub(r"\$\$(.+?)\$\$", r"\\[\1\\]", text, flags=re.DOTALL)
    inline_converted = re.sub(r"\$([^$\n]+?)\$", r"\\(\1\\)", display_converted)
    return inline_converted


def format_card_field_html(text: str) -> str:
    math_converted = convert_markdown_math_to_anki(text)
    escaped = html.escape(math_converted)
    escaped = escaped.replace("\r\n", "\n")
    return escaped.replace("\n", "<br/>")


def create_anki_deck(
    deck_name: str,
    flashcards: list[dict[str, Any]],
    output_file_path: Path
) -> int:
    model = genanki.Model(
        MODEL_ID,
        "LectureLens Basic",
        fields=[{"name": "Question"}, {"name": "Answer"}],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Question}}",
                "afmt": "{{FrontSide}}<hr id=\"answer\">{{Answer}}"
            }
        ],
        css=".card { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 18px; text-align: left; color: rgb(31, 41, 55); line-height: 1.6; padding: 20px; } hr { border: 0; height: 1px; background: rgb(229, 231, 235); margin: 20px 0; } strong { color: rgb(17, 24, 39); } code { background: rgb(243, 244, 246); padding: 2px 4px; border-radius: 4px; font-family: monospace; font-size: 0.9em; }"
    )
    deck = genanki.Deck(compute_stable_deck_id(deck_name), deck_name)

    added_count = 0
    seen_questions: set[str] = set()
    for flashcard in flashcards:
        question = str(flashcard.get("question", "")).strip()
        answer = str(flashcard.get("answer", "")).strip()
        if len(question) == 0 or len(answer) == 0:
            continue
        if question in seen_questions:
            continue
        seen_questions.add(question)
        note = genanki.Note(
            model=model,
            fields=[format_card_field_html(question), format_card_field_html(answer)],
            guid=genanki.guid_for(deck_name, question)
        )
        deck.add_note(note)
        added_count += 1

    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    genanki.Package(deck).write_to_file(str(output_path))
    return added_count
