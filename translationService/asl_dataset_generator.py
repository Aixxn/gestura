import argparse
import ast
import json
import random
from pathlib import Path


PROMPT_PREFIX = "translate ASL gloss to English:"
DEFAULT_SELECTED_EXTRACTION_PATH = (
    Path(__file__).resolve().parents[1] / "model" / "Selected_extraction.py"
)


def load_asl_mini_words(path: Path = DEFAULT_SELECTED_EXTRACTION_PATH) -> set[str]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "ASL_MINI_WORDS" in names:
                value = ast.literal_eval(node.value)
                return {str(word).upper() for word in value}
    raise ValueError(f"ASL_MINI_WORDS not found in {path}")


def _pair(gloss: str, target: str) -> dict[str, str]:
    return {
        "input": f"{PROMPT_PREFIX} {gloss}",
        "target": target,
    }


def _article(noun: str) -> str:
    if noun in {"APPLE"}:
        return "an apple"
    if noun in {"FRIEND"}:
        return "a friend"
    return noun.lower()


def _subject(token: str) -> str:
    return {
        "ME": "I",
        "YOU": "you",
        "HE": "he",
        "WE": "we",
        "THEY": "they",
        "MOTHER": "mother",
        "FRIEND": "friend",
    }[token]


def _be_verb(subject: str) -> str:
    if subject == "I":
        return "am"
    if subject in {"you", "we", "they"}:
        return "are"
    return "is"


def _present_verb(subject: str, verb: str) -> str:
    if subject in {"he", "mother", "friend"}:
        if verb == "go":
            return "goes"
        return f"{verb}s"
    return verb


def _sentence(text: str) -> str:
    return text[0].upper() + text[1:] + "."


def generate_pairs(words: set[str]) -> list[dict[str, str]]:
    words = {word.upper() for word in words}
    pairs: list[dict[str, str]] = []

    subjects = [word for word in ["ME", "YOU", "HE", "WE", "THEY"] if word in words]
    people = [word for word in ["MOTHER", "FRIEND"] if word in words]
    drink_objects = [word for word in ["WATER", "MILK"] if word in words]
    eat_objects = [word for word in ["APPLE"] if word in words]
    thing_objects = [word for word in ["WATER", "MILK", "APPLE"] if word in words]
    adjectives = [word for word in ["HAPPY", "ANGRY", "TIRED", "GOOD", "BAD"] if word in words]

    verb_objects = {
        "DRINK": ("drink", drink_objects),
        "EAT": ("eat", eat_objects),
        "LIKE": ("like", thing_objects + people),
        "WANT": ("want", thing_objects),
        "SEE": ("see", people),
        "ASK": ("ask", people),
    }
    available_action_verbs = {
        gloss: (english, objects)
        for gloss, (english, objects) in verb_objects.items()
        if gloss in words and objects
    }

    for subject_token in subjects:
        subject = _subject(subject_token)

        for adjective in adjectives:
            pairs.append(_pair(
                f"{subject_token} {adjective}",
                _sentence(f"{subject} {_be_verb(subject)} {adjective.lower()}"),
            ))

        for verb_token, (verb, allowed_objects) in available_action_verbs.items():
            for obj in allowed_objects:
                english_verb = _present_verb(subject, verb)
                pairs.append(_pair(
                    f"{subject_token} {verb_token} {obj}",
                    _sentence(f"{subject} {english_verb} {_article(obj)}"),
                ))

    polite_command_verbs = {"DRINK", "EAT", "SEE", "ASK"}
    if "PLEASE" in words:
        for verb_token, (verb, allowed_objects) in available_action_verbs.items():
            if verb_token not in polite_command_verbs:
                continue
            for obj in allowed_objects:
                pairs.append(_pair(
                    f"PLEASE {verb_token} {obj}",
                    _sentence(f"please {verb} {_article(obj)}"),
                ))
                if "HOME" in words:
                    pairs.append(_pair(
                        f"PLEASE {verb_token} HOME {obj}",
                        _sentence(f"please {verb} {_article(obj)} at home"),
                    ))

    if {"WHAT", "YOU", "WANT"}.issubset(words):
        pairs.append(_pair("WHAT YOU WANT", "What do you want?"))

    if {"WHERE", "MOTHER", "GO"}.issubset(words):
        pairs.append(_pair("WHERE MOTHER GO", "Where does mother go?"))

    if {"YOU", "LIKE"}.issubset(words):
        for obj in thing_objects:
            pairs.append(_pair(
                f"YOU LIKE {obj}",
                f"Do you like {_article(obj)}?",
            ))

    if {"MY", "NAME"}.issubset(words):
        pairs.append(_pair("MY NAME", "My name is Gestura."))

    if {"YES"}.issubset(words):
        pairs.append(_pair("YES", "Yes."))

    if {"NO"}.issubset(words):
        pairs.append(_pair("NO", "No."))

    unique: dict[str, dict[str, str]] = {}
    for pair in pairs:
        unique[pair["input"]] = pair
    return list(unique.values())


def write_dataset(
    pairs: list[dict[str, str]],
    output_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Path, Path]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_ratio))
    val_rows = shuffled[:val_size]
    train_rows = shuffled[val_size:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "asl_gloss_train.jsonl"
    val_path = output_dir / "asl_gloss_val.jsonl"

    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    return train_path, val_path


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    content = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text(f"{content}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    words = load_asl_mini_words()
    pairs = generate_pairs(words)
    train_path, val_path = write_dataset(
        pairs,
        args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Wrote {train_path}")
    print(f"Wrote {val_path}")
    print(f"Pairs: {len(pairs)}")


if __name__ == "__main__":
    main()
