class CharacterProcessor:
    def __init__(self):
        self.alphabet = (
            "aąbcćdeęfghijklłmnńoópqrsśtuvwxyzźżAĄBCĆDEĘFGHIJKLŁMNŃOÓPQRSŚTUVWXYZŹŻ !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        )
        self.char_id = {char: idx for idx, char in enumerate(self.alphabet)}
        self.char_id["<PAD>"] = len(self.char_id)
        self.char_id["<UNK>"] = len(self.char_id)

        self.id_char = {idx: char for char, idx in self.char_id.items()}
        self.vocab_size = len(self.char_id)
        self.pad_token_id = self.char_id["<PAD>"]
        self.unk_token_id = self.char_id["<UNK>"]

    def text_to_sequence(self, text):
        return [self.char_id.get(char, self.unk_token_id) for char in text]

    def sequence_to_text(self, sequence):
        return "".join([self.id_char.get(idx, "") for idx in sequence if idx != self.pad_token_id])
