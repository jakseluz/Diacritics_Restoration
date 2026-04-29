import torch
import torch.nn as nn
from .processor import CharacterProcessor


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

    def restore(self, text):
        self.model.eval()
        original_length = len(text)
        text = text.translate(self.trans_table)
        tokens = self.processor.text_to_sequence(text)

        if len(tokens) < self.window_size:
            padding_len = self.window_size - len(tokens)
            tokens += [self.processor.pad_token_id] * padding_len

        input_tensor = torch.tensor(tokens[: self.window_size]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            predicted_ids = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        restored_text = self.processor.sequence_to_text(predicted_ids[:original_length])

        return restored_text
