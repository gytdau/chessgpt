import numpy as np
import os
import pickle

# Assign ints for all
stoi = {
    pos: idx
    for idx, pos in enumerate([f"{chr(97+i)}{j+1}" for i in range(8) for j in range(8)])
}

# Add spaces and promotion mappings
new_tokens = [" ", "q", "r", "b", "n", "\n"]
for token in new_tokens:
    stoi[token] = len(stoi)

vocab_size = len(stoi.items())

itos = {idx: pos for pos, idx in stoi.items()}


def encode(uci_line):
    # Define the mapping for board positions to integers
    tokens = []
    idx = 0
    while idx < len(uci_line):
        # If current and next characters form a board position, tokenize it
        if uci_line[idx : idx + 2] in stoi:
            tokens.append(stoi[uci_line[idx : idx + 2]])
            idx += 2
        # If it's a promotion character, tokenize it
        elif uci_line[idx] in ["q", "r", "b", "n"]:
            tokens.append(stoi[uci_line[idx]])
            idx += 1
        # If it's a space, tokenize it
        elif uci_line[idx] == " ":
            tokens.append(stoi[" "])
            idx += 1
        # If it's a newline, tokenize it
        elif uci_line[idx] == "\n":
            tokens.append(stoi["\n"])
            idx += 1
    return tokens


def decode(tokens):
    # Reverse mapping

    uci_string = ""
    for token in tokens:
        uci_string += itos[token]
    return uci_string


# Test the encode and decode functions
def test_uci_conversion():
    test_string = "g1f3 g8f6 d2d4 d7d5q"
    encoded = encode(test_string)
    decoded = decode(encoded)
    assert test_string == decoded, f"Expected {test_string}, but got {decoded}"
    print("Test passed!")


def convert_log():
    with open("log.log", "r") as f:
        encoded = encode(f.read())

        train = encoded[: int(len(encoded) * 0.9)]
        val = encoded[int(len(encoded) * 0.9) :]

        train_ids = np.array(train, dtype=np.uint16)
        val_ids = np.array(val, dtype=np.uint16)
        train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
        val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }

    with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    test_uci_conversion()
    convert_log()
