"""Ask the model every interaction question once and store the answers (Person 5).

Run on a computer where the model is set up (full requirements.txt, the search
index built from the shared CIMA file, GROQ_API_KEY in .env), from the project
root:

    python -m src.guardar_interacciones

It writes data/interacciones_guardadas.json with one answer per (technical
sheet, ingredient) search for the medicines in data/catalogo_medicamentos.json.
Upload that file to GitHub: the shared link, and any computer without the
model, then shows these stored answers instead of "Not checked".

The answers are saved after each question. Running the command again only asks
the questions that are missing, so it can be repeated after an interruption or
an API limit. Use --start-over after the CIMA file or the medicine list changes.
"""
from __future__ import annotations

import argparse

from src.main import (
    STORED_ANSWERS_FILE,
    all_questions,
    load_catalogue,
    model_status,
    save_answers,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--pause", type=float, default=2.0,
        help="seconds to wait after each question that reached the model (default 2, "
             "to stay under Groq's free-tier limit)",
    )
    parser.add_argument(
        "--start-over", action="store_true",
        help="delete the stored answers first (after the CIMA file or the medicine list changes)",
    )
    args = parser.parse_args(argv)

    ready, message = model_status()
    if not ready:
        print(f"The model is not available on this computer: {message}")
        return 1
    from src.Rag.rag import review_interaction

    if args.start_over and STORED_ANSWERS_FILE.exists():
        STORED_ANSWERS_FILE.unlink()

    questions = all_questions(load_catalogue())
    print(f"{len(questions)} searches to store in {STORED_ANSWERS_FILE}.")
    counts = save_answers(STORED_ANSWERS_FILE, review_interaction, questions, pause=args.pause)
    print(
        f"\nDone: {counts['kept']} new answers stored, {counts['skipped']} already stored, "
        f"{counts['retry_later']} to retry."
    )
    if counts["retry_later"]:
        print("Run the same command again in a few minutes to retry those.")
    else:
        print("Upload data/interacciones_guardadas.json to GitHub.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
