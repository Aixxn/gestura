import json

import asl_dataset_generator as gen


def test_load_asl_mini_words_from_selected_extraction():
    words = gen.load_asl_mini_words()

    assert "PLEASE" in words
    assert "DRINK" in words
    assert "WATER" in words
    assert "APPLE" in words


def test_generate_pairs_contains_core_asl_mini_examples():
    pairs = gen.generate_pairs({
        "ME", "YOU", "PLEASE", "DRINK", "EAT", "WATER", "MILK", "APPLE",
        "HOME", "LIKE", "WHAT", "WANT", "WHERE", "MOTHER", "GO", "TIRED",
    })

    pair_lookup = {pair["input"]: pair["target"] for pair in pairs}

    assert pair_lookup["translate ASL gloss to English: ME DRINK WATER"] == "I drink water."
    assert pair_lookup["translate ASL gloss to English: ME EAT APPLE"] == "I eat an apple."
    assert pair_lookup["translate ASL gloss to English: PLEASE DRINK HOME MILK"] == "Please drink milk at home."
    assert pair_lookup["translate ASL gloss to English: WHAT YOU WANT"] == "What do you want?"
    assert pair_lookup["translate ASL gloss to English: WHERE MOTHER GO"] == "Where does mother go?"
    assert pair_lookup["translate ASL gloss to English: ME TIRED"] == "I am tired."


def test_generate_pairs_avoids_invalid_verb_object_pairs():
    pairs = gen.generate_pairs({
        "ME", "WE", "PLEASE", "DRINK", "EAT", "LIKE", "SEE", "WATER", "MILK", "APPLE",
        "FRIEND", "MOTHER", "HOME",
    })
    inputs = {pair["input"] for pair in pairs}

    assert "translate ASL gloss to English: ME EAT FRIEND" not in inputs
    assert "translate ASL gloss to English: WE DRINK FRIEND" not in inputs
    assert "translate ASL gloss to English: PLEASE EAT WATER" not in inputs
    assert "translate ASL gloss to English: PLEASE LIKE FRIEND" not in inputs
    assert "translate ASL gloss to English: PLEASE LIKE HOME APPLE" not in inputs
    assert "translate ASL gloss to English: ME SEE FRIEND" in inputs
    assert "translate ASL gloss to English: PLEASE DRINK WATER" in inputs
    assert "translate ASL gloss to English: PLEASE EAT APPLE" in inputs


def test_write_dataset_splits_jsonl(tmp_path):
    pairs = [
        {"input": "translate ASL gloss to English: ME DRINK WATER", "target": "I drink water."},
        {"input": "translate ASL gloss to English: ME EAT APPLE", "target": "I eat an apple."},
        {"input": "translate ASL gloss to English: PLEASE DRINK WATER", "target": "Please drink water."},
        {"input": "translate ASL gloss to English: YOU LIKE APPLE", "target": "Do you like an apple?"},
    ]

    train_path, val_path = gen.write_dataset(pairs, tmp_path, val_ratio=0.25, seed=7)

    train_rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    val_rows = [json.loads(line) for line in val_path.read_text().splitlines()]

    assert len(train_rows) == 3
    assert len(val_rows) == 1
    assert all(set(row) == {"input", "target"} for row in train_rows + val_rows)
