import random
from pathlib import Path
from typing import Any
import genanki


ANKI_MODEL_ID = 1607392319
ANKI_DECK_ID_BASE = 2059400110


def get_default_flashcard_model() -> genanki.Model:
    model = genanki.Model(
        ANKI_MODEL_ID,
        "LectureLens Basic Flashcard",
        fields=[
            {"name": "Question"},
            {"name": "Answer"}
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "<div style='font-family: Arial; font-size: 20px; text-align: center; color: #2D3748;'>{{Question}}</div>",
                "afmt": "<div style='font-family: Arial; font-size: 20px; text-align: center; color: #2D3748;'>{{FrontSide}}<hr id='answer'><div style='color: #2B6CB0; font-weight: bold;'>{{Answer}}</div></div>"
            }
        ],
        css=".card { font-family: arial; font-size: 20px; text-align: center; color: black; background-color: white; }"
    )
    return model


def create_anki_deck(
    deck_name: str,
    flashcards: list[dict[str, str]],
    output_file_path: Path
) -> None:
    deck_id = ANKI_DECK_ID_BASE + (hash(deck_name) % 1000000)
    deck = genanki.Deck(deck_id, deck_name)
    model = get_default_flashcard_model()

    for card in flashcards:
        question = card.get("question", "").strip()
        answer = card.get("answer", "").strip()
        if len(question) == 0 or len(answer) == 0:
            continue

        note = genanki.Note(
            model=model,
            fields=[question, answer]
        )
        deck.add_note(note)

    package = genanki.Package(deck)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    package.write_to_file(str(output_file_path))
