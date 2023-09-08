# %%
# Dataset prep

with open("input.txt", "r") as f:
    dataset = f.read()

vocab = list(sorted(list(set(dataset))))

vocab_size = len(vocab)


def encode(text):
    return [vocab.index(c) for c in text]


def decode(indices):
    return "".join([vocab[i] for i in indices])


dataset = torch.tensor(encode(dataset), dtype=torch.long)
# %%
# 10% of the dataset is used for validation
val_size = int(len(dataset) * 0.1)
train_dataset, val_dataset = dataset[:-val_size], dataset[-val_size:]
train_dataset = train_dataset.to("cuda")
val_dataset = val_dataset.to("cuda")

batch_size = 256  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?


def get_batch(split):
    # generate a small batch of data of inputs x and targets ya
    data = train_dataset if split == "train" else val_dataset
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# %%

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import wandb

torch.manual_seed(1337)

embedding_dimensions = 64
n_head = 4
dropout = 0.2


class MaskedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # self.keys = nn.Linear(embedding_dimensions, embedding_dimensions)
        # self.queries = nn.Linear(embedding_dimensions, embedding_dimensions)
        # self.values = nn.Linear(embedding_dimensions, embedding_dimensions)

        self.keys_queries_and_values = nn.Linear(
            embedding_dimensions, 3 * embedding_dimensions
        )

        self.output_projection = nn.Linear(embedding_dimensions, embedding_dimensions)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # unpack:
        embedded = self.keys_queries_and_values(x)  # (B, T, 3 * C)

        keys, queries, values = embedded.split(
            embedding_dimensions, dim=-1
        )  # (B, T, C)

        keys = keys.view(
            -1, block_size, n_head, embedding_dimensions // n_head
        ).transpose(
            1, 2
        )  # (B, n_head, T, C // n_head)
        queries = queries.view(
            -1, block_size, n_head, embedding_dimensions // n_head
        ).transpose(
            1, 2
        )  # (B, n_head, T, C // n_head)
        values = values.view(
            -1, block_size, n_head, embedding_dimensions // n_head
        ).transpose(
            1, 2
        )  # (B, n_head, T, C // n_head)

        # keys = keys.transpose(1, 2)  # (B, C, T)
        # queries = queries.transpose(1, 2)  # (B, C, T)
        # values = values.transpose(1, 2)  # (B, C, T)

        # Compute the self-attention
        # (B, n_head, T, C // n_head) x (B, n_head, C // n_head, T) -> (B, n_head, T, T)
        logits = queries @ keys.transpose(2, 3)
        mystery = 1.0 / math.sqrt(keys.size(-1))  # is this a regularisation loss?
        logits = logits * mystery

        # Mask out the lower half of the scores, so we can't look into the future

        logits = logits.masked_fill(self.mask == 0, float("-inf"))

        # Turn the scores into probabilities
        probs = F.softmax(logits, dim=-1)

        # probs = self.attn_dropout(probs)

        y = (
            probs @ values
        )  # (B, n_head, T, T) x (B, n_head, T, C // n_head) -> (B, n_head, T, C // n_head)

        y = y.transpose(1, 2).contiguous()  # (B, T, n_head, C // n_head)
        y = y.view(-1, block_size, embedding_dimensions)  # (B, T, C)

        # y = self.resid_dropout(y)

        # output projection does not seem to help
        # y = self.output_projection(y)

        return y


class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = MaskedAttention()
        self.layer_norm = nn.LayerNorm(embedding_dimensions)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dimensions, 4 * embedding_dimensions),
            nn.GELU(),
            nn.Linear(4 * embedding_dimensions, embedding_dimensions),
        )
        self.layer_norm_2 = nn.LayerNorm(embedding_dimensions)

    def forward(self, x):
        y = self.attention(self.layer_norm(x))
        x = x + y
        y = self.mlp(self.layer_norm_2(x))
        x = x + y
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dimensions)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dimensions)
        self.language_modelling_head = nn.Linear(embedding_dimensions, vocab_size)

        self.transformer = nn.ModuleDict(
            {
                "h": nn.ModuleList([Block() for _ in range(8)]),
            }
        )

    def forward(self, idx, targets=None):
        token_embeddings = self.token_embedding_table(
            idx
        )  # (batch, block_size, embedding_dim)
        # Surely there is a more efficient way to do this, since we are taking the entire position embedding table:
        position_embeddings = self.position_embedding_table(
            torch.arange(block_size).to(idx.device)
        )  # (block_size, embedding_dim)

        # Add the position embeddings to the token embeddings
        x = token_embeddings + position_embeddings  # (batch, block_size, embedding_dim)

        # Pass through the model
        for block in self.transformer.h:
            x = block(x)

        logits = self.language_modelling_head(x)  # (batch, block_size, vocab_size)

        B, T, C = logits.shape
        logits_view = logits.view(B * T, C)

        loss = None
        if targets is not None:
            targets = targets.view(-1)

            loss = F.cross_entropy(logits_view, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])

            # last logit
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)

        return idx


model = BigramLanguageModel()
model.to("cuda")


def sample():
    start = torch.zeros(1, block_size, dtype=torch.long, device="cuda")
    out = model.generate(start, 50).to("cpu")
    out = out[0].tolist()
    # print(decode(torch.argmax(out, dim=1)))
    print("---")
    print(decode(out[block_size:]))


sample()


def estimate_loss(steps, times_per_step):
    train_loss = 0

    for _ in range(100):
        x, y = get_batch("train")
        logits, loss = model(x, y)
        train_loss += loss.item()

    train_loss /= 100

    val_loss = 0
    for _ in range(100):
        x, y = get_batch("val")
        logits, loss = model(x, y)
        val_loss += loss.item()

    val_loss /= 100

    time_per_step_us = (sum(times_per_step) / len(times_per_step)) * 1e6

    wandb.log(
        {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "time_per_step": time_per_step_us,
        },
        step=steps,
    )


times_per_step = []

wandb.init(project="nanogpt")
wandb.watch(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for steps in range(100000):
    start = time.time()
    optimizer.zero_grad()

    batches, targets = get_batch("train")  # correct terminology for vars?

    logits, loss = model(batches, targets)
    loss.backward()
    optimizer.step()
    end = time.time()

    times_per_step.append(end - start)
    times_per_step = times_per_step[-100:]

    if steps % 1000 == 0:
        estimate_loss(steps, times_per_step)
        sample()


# %%
start = torch.zeros(1, batch_size, dtype=torch.long)
out = model.generate(start, 50)
out = out[0].tolist()
print(decode(out))
# %%

# toy example

B, T, C = 4, 8, 2
logits = torch.randn(B, T, C)
print(logits.shape)

# %%
