import html
import zlib
from pathlib import Path
from typing import Any

import genanki

MODEL_ID = 1607392319


def compute_stable_deck_id(deck_name: str) -> int:
    checksum = zlib.crc32(deck_name.encode("utf-8"))
    return (1 << 30) + (checksum % (1 << 30))


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
        ]
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
            fields=[html.escape(question), html.escape(answer)],
            guid=genanki.guid_for(question)
        )
        deck.add_note(note)
        added_count += 1

    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    genanki.Package(deck).write_to_file(str(output_path))
    return added_count
