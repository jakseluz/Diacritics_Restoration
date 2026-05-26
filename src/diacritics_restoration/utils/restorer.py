import torch
import torch.nn as nn
from .processor import CharacterProcessor

try:
    from transformers import PreTrainedTokenizerBase, PreTrainedModel
except Exception:
    PreTrainedTokenizerBase = object
    PreTrainedModel = object


class DiacriticsRestorer:
    def __init__(
        self,
        model: nn.Module,
        processor: CharacterProcessor,
        device: torch.device,
        window_size: int = 256,
        overlap: int = 32,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.window_size = window_size
        self.overlap = max(0, int(overlap))

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
        n = len(tokens)

        # If the text is shorter than or equal to the window size, process it in one go
        if n <= self.window_size:
            if n < self.window_size:
                padding_len = self.window_size - n
                tokens += [self.processor.pad_token_id] * padding_len

            input_tensor = (
                torch.tensor(tokens[: self.window_size])
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                logits = self.model(input_tensor)
                predicted_ids = (
                    torch.argmax(logits, dim=1)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .tolist()
                )

            restored_text = self.processor.sequence_to_text(
                predicted_ids[:original_length]
            )
            self.texts_history[original_text] = restored_text
            return restored_text

        # Process the text in overlapping windows
        stride = self.window_size - min(self.overlap, self.window_size - 1)
        out_ids: list[int] = []

        with torch.no_grad():
            start = 0
            first = True
            while start < n:
                end = min(start + self.window_size, n)
                window = tokens[start:end]
                if len(window) < self.window_size:
                    padding_len = self.window_size - len(window)
                    window += [self.processor.pad_token_id] * padding_len

                input_tensor = torch.tensor(window).unsqueeze(0).to(self.device)
                logits = self.model(input_tensor)  # (1, vocab, window)
                pred_window = (
                    torch.argmax(logits, dim=1)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .tolist()
                )

                # sklejanie bez duplikowania nałożeń
                if first:
                    take_from = 0
                    first = False
                else:
                    take_from = min(self.overlap, end - start)

                out_ids.extend(
                    pred_window[
                        take_from : take_from + (end - start - take_from)
                    ]
                )

                start += stride

        # out_ids powinno mieć długość ~= n; zabezpieczenie na wypadek skrajnych parametrów
        if len(out_ids) < n:
            out_ids.extend(tokens[len(out_ids) : n])
        elif len(out_ids) > n:
            out_ids = out_ids[:n]

        restored_text = self.processor.sequence_to_text(
            out_ids[:original_length]
        )
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
        overlap_chars: int = 64,
        prefer_whitespace_split: bool = True,
        generate_kwargs: dict | None = None,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.max_input_length = max_input_length
        self.chunk_chars = chunk_chars

        self.overlap_chars = max(0, int(overlap_chars))
        self.prefer_whitespace_split = bool(prefer_whitespace_split)

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