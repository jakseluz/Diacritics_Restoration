import torch
import torch.nn as nn
from .processor import CharacterProcessor

try:
    from transformers import PreTrainedTokenizerBase, PreTrainedModel
except Exception:
    PreTrainedTokenizerBase = object
    PreTrainedModel = object


class DiacriticsRestorer:
    def __init__(self, model: nn.Module, processor: CharacterProcessor, device: torch.device, window_size: int = 256):
        self.model = model
        self.processor = processor
        self.device = device
        self.window_size = window_size
        mapping = {
            "ą": "a",
            "ć": "c",
            "ę": "e",
            "ł": "l",
            "ń": "n",
            "ó": "o",
            "ś": "s",
            "ź": "z",
            "ż": "z",
            "Ą": "A",
            "Ć": "C",
            "Ę": "E",
            "Ł": "L",
            "Ń": "N",
            "Ó": "O",
            "Ś": "S",
            "Ź": "Z",
            "Ż": "Z",
        }
        self.trans_table = str.maketrans(mapping)
        self.texts_history: dict[str, str] = dict()

    def restore(self, text):
        original_text = text
        original_length = len(original_text)
        self.model.eval()

        src_text = text.translate(self.trans_table)
        tokens = self.processor.text_to_sequence(src_text)

        if len(tokens) < self.window_size:
            padding_len = self.window_size - len(tokens)
            tokens += [self.processor.pad_token_id] * padding_len

        input_tensor = torch.tensor(tokens[: self.window_size]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            predicted_ids = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        restored_text = self.processor.sequence_to_text(predicted_ids[:original_length])

        self.texts_history[original_text] = restored_text

        return restored_text

    def calculate_character_error_rate(self, original_text, restored_text):
        original_chars = list(original_text)
        restored_chars = list(restored_text)

        total_chars = len(original_chars)
        differences = 0
        for o, r in zip(original_chars, restored_chars):
            if o != r:
                differences += 1
        return differences / total_chars if total_chars > 0 else 0.0

    def calculate_history_character_error_rate(self):
        originals = "".join(self.texts_history.keys())
        restores = "".join(self.texts_history.values())
        return self.calculate_character_error_rate(originals, restores)


class ByT5DiacriticsRestorer(DiacriticsRestorer):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        max_input_length: int = 512,
        chunk_chars: int | None = 512,
        generate_kwargs: dict | None = None,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_input_length = max_input_length
        self.chunk_chars = chunk_chars
        self.generate_kwargs = generate_kwargs or {
            "num_beams": 1,
            "do_sample": False,
        }

        mapping = {
            "ą": "a",
            "ć": "c",
            "ę": "e",
            "ł": "l",
            "ń": "n",
            "ó": "o",
            "ś": "s",
            "ź": "z",
            "ż": "z",
            "Ą": "A",
            "Ć": "C",
            "Ę": "E",
            "Ł": "L",
            "Ń": "N",
            "Ó": "O",
            "Ś": "S",
            "Ź": "Z",
            "Ż": "Z",
        }
        self.trans_table = str.maketrans(mapping)
        self.texts_history: dict[str, str] = {}

    def _chunks(self, text: str) -> list[str]:
        if not self.chunk_chars or len(text) <= self.chunk_chars:
            return [text]
        return [
            text[i : i + self.chunk_chars]
            for i in range(0, len(text), self.chunk_chars)
        ]

    @torch.no_grad()
    def restore(self, text: str) -> str:
        original_text = text
        self.model.eval()

        src_text = text.translate(self.trans_table)

        outputs: list[str] = []
        for chunk in self._chunks(src_text):
            sequence = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length,
            )
            sequence = {k: v.to(self.device) for k, v in sequence.items()}

            gen_ids = self.model.generate(**sequence, **self.generate_kwargs)

            output_text = self.tokenizer.decode(
                gen_ids[0], skip_special_tokens=True
            )
            outputs.append(output_text)

        restored_text = "".join(outputs)
        self.texts_history[original_text] = restored_text
        return restored_text