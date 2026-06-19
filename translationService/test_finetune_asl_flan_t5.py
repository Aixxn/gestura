import json

import torch

import finetune_asl_flan_t5 as ft


class FakeTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text,
        max_length,
        padding,
        truncation,
        return_tensors,
    ):
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        values = [len(token) for token in text.split()][:max_length]
        values = values + [self.pad_token_id] * (max_length - len(values))
        return {
            "input_ids": torch.tensor([values]),
            "attention_mask": torch.tensor([[1 if value else 0 for value in values]]),
        }


def test_load_jsonl_reads_training_pairs(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"input": "translate ASL gloss to English: ME DRINK WATER", "target": "I drink water."}),
            json.dumps({"input": "translate ASL gloss to English: ME EAT APPLE", "target": "I eat an apple."}),
        ])
        + "\n"
    )

    rows = ft.load_jsonl(path)

    assert rows == [
        {"input": "translate ASL gloss to English: ME DRINK WATER", "target": "I drink water."},
        {"input": "translate ASL gloss to English: ME EAT APPLE", "target": "I eat an apple."},
    ]


def test_asl_gloss_dataset_tokenizes_and_masks_padding_labels():
    rows = [
        {"input": "translate ASL gloss to English: ME DRINK WATER", "target": "I drink water."},
    ]

    dataset = ft.AslGlossDataset(
        rows,
        FakeTokenizer(),
        max_input_length=8,
        max_target_length=5,
    )

    item = dataset[0]

    assert set(item) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].shape == torch.Size([8])
    assert item["attention_mask"].shape == torch.Size([8])
    assert item["labels"].shape == torch.Size([5])
    assert -100 in item["labels"].tolist()
