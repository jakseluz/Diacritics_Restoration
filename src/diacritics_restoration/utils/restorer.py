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

    def _normalize_no_diacritics(self, text: str) -> str:
        return text.translate(self.trans_table)

    def _iter_spans(self, text: str) -> list[tuple[int, int]]:
        n = len(text)
        if not self.chunk_chars or n <= self.chunk_chars:
            return [(0, n)]

        spans: list[tuple[int, int]] = []
        start = 0
        while start < n:
            hard_end = min(start + self.chunk_chars, n)
            end = hard_end

            if self.prefer_whitespace_split and hard_end < n:
                window = text[start:hard_end]
                last_ws = max(
                    window.rfind(" "), window.rfind("\n"), window.rfind("\t")
                )
                if last_ws > max(10, int(0.6 * len(window))):
                    end = start + last_ws + 1

            spans.append((start, end))
            if end >= n:
                break

            start = max(end - self.overlap_chars, start + 1)

        return spans

    def _trim_overlap_from_output(
        self, output: str, overlap_src_no_diacritics: str
    ) -> str:
        if not overlap_src_no_diacritics:
            return output

        normalized_output = self._normalize_no_diacritics(output)
        search = normalized_output[
            : max(128, 3 * len(overlap_src_no_diacritics))
        ]
        position = search.rfind(overlap_src_no_diacritics)
        if position != -1:
            cut_point = position + len(overlap_src_no_diacritics)
            return output[cut_point:]

        if self.overlap_chars > 0 and len(output) > self.overlap_chars:
            return output[self.overlap_chars :]

        return output

    @torch.no_grad()
    def restore(self, text: str) -> str:
        original_text = text
        self.model.eval()

        if not text:
            self.texts_history[original_text] = text
            return text

        src_text = text.translate(self.trans_table)
        spans = self._iter_spans(src_text)

        outputs: list[str] = []
        previous_end = 0

        for start, end in spans:
            chunk = src_text[start:end]

            sequence = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length,
            )
            sequence = {k: v.to(self.device) for k, v in sequence.items()}

            gen_kwargs = dict(self.generate_kwargs)

            if (
                "max_new_tokens" not in gen_kwargs
                and "max_length" not in gen_kwargs
            ):
                in_len = int(sequence["input_ids"].shape[1])
                gen_kwargs["max_new_tokens"] = in_len + 16

            gen_ids = self.model.generate(**sequence, **gen_kwargs)

            output_text = self.tokenizer.decode(
                gen_ids[0], skip_special_tokens=True
            )

            if outputs and start < previous_end:
                overlap_src = src_text[start:previous_end]
                output_text = self._trim_overlap_from_output(
                    output_text, overlap_src
                )

            outputs.append(output_text)
            previous_end = end

        restored_text = "".join(outputs)

        if len(restored_text) < len(original_text):
            restored_text += original_text[len(restored_text) :]
        elif len(restored_text) > len(original_text):
            restored_text = restored_text[: len(original_text)]

        self.texts_history[original_text] = restored_text
        return restored_text