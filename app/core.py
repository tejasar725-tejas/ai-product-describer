"""A tiny byte-level GPT for learning product-description generation from scratch."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

# Original, hand-written starter corpus for the demonstration category.
SEED_PRODUCTS = [
    'Product: Everyday Canvas Tote\nFeatures: cotton canvas, roomy interior, reinforced handles, inner pocket\nTone: friendly\nDescription: Meet the Everyday Canvas Tote, a roomy cotton carryall with reinforced handles and an inside pocket. Its simple design keeps books, groceries and daily essentials close at hand.',
    'Product: Meadow Print Tote\nFeatures: sturdy cotton canvas, botanical print, comfortable shoulder straps\nTone: cheerful\nDescription: Add a little colour to your routine with the Meadow Print Tote. Its cheerful botanical pattern and comfortable straps make everyday carrying easy.',
    'Product: Market Day Shopper\nFeatures: wide opening, durable stitching, lightweight construction\nTone: practical\nDescription: The Market Day Shopper is a practical companion for busy days. Its wide opening and durable stitching make space for your everyday finds.',
    'Product: Minimalist Zip Pouch\nFeatures: smooth zip closure, compact shape, soft fabric\nTone: simple\nDescription: Keep small items together in the Minimalist Zip Pouch. Its compact shape slips easily into a backpack or handbag.',
    'Product: Soft Cotton Storage Basket\nFeatures: flexible woven body, soft cotton, useful storage space\nTone: warm\nDescription: Organise your desk with this Soft Cotton Storage Basket. It holds stationery, accessories and craft supplies while bringing a neat finish to shelves.',
    'Product: Little Things Gift Pouch\nFeatures: natural cotton texture, drawstring closure, compact size\nTone: thoughtful\nDescription: Designed for thoughtful gifting, the Little Things Gift Pouch keeps jewellery, keepsakes and small treats together.',
    'Product: Weekend Utility Tote\nFeatures: generous main compartment, sturdy handles, clean design\nTone: friendly\nDescription: The Weekend Utility Tote offers a generous compartment and sturdy handles for errands, classes and short outings.',
    'Product: Pocket Canvas Bag\nFeatures: durable canvas, front slip pocket, everyday size\nTone: practical\nDescription: Keep daily essentials within reach with the Pocket Canvas Bag. Its front pocket offers quick access to smaller items.',
    'Product: Striped Cotton Shopper\nFeatures: classic stripes, roomy interior, reinforced seams\nTone: classic\nDescription: Bring an easy feel to your everyday carry with the Striped Cotton Shopper. Its roomy interior is ready for books and shopping.',
    'Product: Travel Organiser Pouch\nFeatures: lightweight fabric, smooth zip, organised compartments\nTone: useful\nDescription: The Travel Organiser Pouch helps keep cables, toiletries and small accessories together. Its lightweight body makes packing simple.',
    'Product: Natural Storage Tote\nFeatures: flexible shape, practical handles, understated finish\nTone: calm\nDescription: The Natural Storage Tote offers flexible space for toys, towels or craft materials. Its understated finish blends easily with different rooms.',
    'Product: Floral Mini Pouch\nFeatures: floral print, compact body, secure zip\nTone: cheerful\nDescription: The Floral Mini Pouch is sized for the little things you do not want to lose. Store coins, cards or earbuds in its bright printed finish.',
    'Product: Foldable Cotton Shopper\nFeatures: foldable design, lightweight cotton, comfortable handles\nTone: practical\nDescription: Make market trips simpler with the Foldable Cotton Shopper. It packs down when not in use and opens into a useful everyday bag.',
    'Product: Classic Canvas Lunch Bag\nFeatures: easy-clean lining, secure top, lightweight body\nTone: simple\nDescription: The Classic Canvas Lunch Bag offers a simple way to carry a meal and a few snacks. Its easy-clean lining suits everyday routines.',
    'Product: Artist Supply Pouch\nFeatures: broad zip opening, durable fabric, roomy interior\nTone: creative\nDescription: Keep creative tools together in the Artist Supply Pouch. Its broad zip opening makes contents easy to see.',
    'Product: Everyday Essentials Kit\nFeatures: compact pouch, lightweight design, neat shape\nTone: friendly\nDescription: The Everyday Essentials Kit keeps small belongings in one place. Its neat shape fits into larger bags without taking up much room.',
    'Product: Cotton Rope Basket\nFeatures: soft structure, practical handles, woven cotton\nTone: warm\nDescription: Give your shelves a tidy refresh with the Cotton Rope Basket. Its soft structure and handles make it easy to move between rooms.',
    'Product: Simple Stripe Pencil Case\nFeatures: stripe pattern, dependable zip, lightweight design\nTone: simple\nDescription: The Simple Stripe Pencil Case keeps pens and pencils organised for school, work or sketching.',
    'Product: Compact Travel Pouch\nFeatures: compact size, lightweight fabric, secure zip\nTone: useful\nDescription: Pack light with the Compact Travel Pouch. It keeps chargers, cosmetics and other small essentials in one place.',
    'Product: Botanical Canvas Shopper\nFeatures: botanical print, spacious interior, durable canvas\nTone: cheerful\nDescription: The Botanical Canvas Shopper brings a fresh printed detail to a practical everyday bag. Its spacious interior is ready for books and shopping.'
]


def milestone_1() -> list[str]:
    """Create/load the small, category-specific corpus and return descriptions."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "product_descriptions.txt"
    if not path.exists():
        path.write_text("\n\n".join(SEED_PRODUCTS) + "\n", encoding="utf-8")
    descriptions = [x.strip() for x in path.read_text(encoding="utf-8").split("\n\n") if x.strip()]
    if len(descriptions) < 5:
        raise ValueError("The corpus needs at least five product descriptions.")
    return descriptions


class BPETokenizer:
    """A compact byte-level BPE tokenizer. Base vocabulary is all 256 byte values."""
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = list(merges or [])
        self.pair_to_id = {pair: 256 + i for i, pair in enumerate(self.merges)}
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for i, (left, right) in enumerate(self.merges):
            self.id_to_bytes[256 + i] = self.id_to_bytes[left] + self.id_to_bytes[right]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @classmethod
    def train(cls, texts: Iterable[str], num_merges: int = 100) -> "BPETokenizer":
        sequences = [[b for b in text.encode("utf-8")] for text in texts]
        merges: list[tuple[int, int]] = []
        for _ in range(max(0, num_merges)):
            counts: Counter = Counter()
            for seq in sequences:
                counts.update(zip(seq, seq[1:]))
            if not counts:
                break
            pair, frequency = counts.most_common(1)[0]
            if frequency < 2:
                break
            new_id = 256 + len(merges)
            merged_sequences = []
            for seq in sequences:
                out, i = [], 0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                merged_sequences.append(out)
            sequences = merged_sequences
            merges.append(pair)
        return cls(merges)

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        for pair in self.merges:
            new_id = self.pair_to_id[pair]
            out, i = [], 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
        return tokens

    def decode(self, ids: Iterable[int]) -> str:
        raw = b"".join(self.id_to_bytes[int(i)] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"merges": self.merges}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return cls([tuple(pair) for pair in obj["merges"]])


def milestone_2(texts: list[str] | None = None, num_merges: int = 100) -> BPETokenizer:
    texts = texts or milestone_1()
    tokenizer = BPETokenizer.train(texts, num_merges=num_merges)
    tokenizer.save(MODEL_DIR / "tokenizer.json")
    return tokenizer


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        head_dim = channels // self.n_head
        q = self.query(x).view(batch, length, self.n_head, head_dim).transpose(1, 2)
        k = self.key(x).view(batch, length, self.n_head, head_dim).transpose(1, 2)
        v = self.value(x).view(batch, length, self.n_head, head_dim).transpose(1, 2)
        weights = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        weights = weights.masked_fill(self.causal_mask[:length, :length] == 0, float("-inf"))
        weights = self.attn_dropout(F.softmax(weights, dim=-1))
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(config.n_embd, config.n_head, config.block_size, config.dropout) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch, length = idx.shape
        if length > self.config.block_size:
            raise ValueError("Input is longer than the model context window")
        positions = torch.arange(length, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = 40):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def milestone_3(vocab_size: int = 356, block_size: int = 128) -> GPTLanguageModel:
    """Assemble a small GPT; vocab_size should match the tokenizer."""
    return GPTLanguageModel(GPTConfig(vocab_size=vocab_size, block_size=block_size))


def _get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    if len(data) <= block_size + 1:
        data = data.repeat((block_size + 2) // len(data) + 1)
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x, y


def milestone_4(texts: list[str] | None = None, epochs: int = 250, num_merges: int = 100,
                learning_rate: float = 3e-3, seed: int = 42) -> dict:
    """Train GPT from random initialization and save reproducible artifacts."""
    random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        torch.set_num_threads(1)
    texts = texts or milestone_1()
    tokenizer = BPETokenizer.train(texts, num_merges)
    encoded = [tokenizer.encode(t + "\n\n") for t in texts]
    flat = [token for row in encoded for token in row]
    data = torch.tensor(flat, dtype=torch.long)
    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = GPTLanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    model.train()
    for epoch in range(epochs):
        xb, yb = _get_batch(data, config.block_size, batch_size=16)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            losses.append({"step": epoch + 1, "loss": round(float(loss.item()), 5)})
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(MODEL_DIR / "tokenizer.json")
    torch.save({"config": asdict(config), "model": model.state_dict()}, MODEL_DIR / "product_gpt.pt")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"architecture": "GPT decoder-only transformer", "training_examples": len(texts), "tokens": len(flat),
              "vocab_size": tokenizer.vocab_size, "epochs": epochs, "seed": seed, "loss_history": losses,
              "limitations": "Small demonstration corpus; outputs are experimental and should be reviewed before commercial use."}
    (OUTPUT_DIR / "training_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_model(model_path: Path | None = None):
    model_path = model_path or MODEL_DIR / "product_gpt.pt"
    tokenizer_path = model_path.parent / "tokenizer.json"
    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError("No trained model found. Run: python -m app.cli train")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    config = GPTConfig(**checkpoint["config"])
    model = GPTLanguageModel(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, BPETokenizer.load(tokenizer_path)


def milestone_5(prompt: str, max_new_tokens: int = 180, temperature: float = 0.8) -> str:
    model, tokenizer = load_model()
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = tokenizer.encode("Product description: ")
    context = torch.tensor([prompt_ids[-model.config.block_size:]], dtype=torch.long)
    output = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature)
    generated = tokenizer.decode(output[0].tolist())
    result = generated[len(tokenizer.decode(context[0].tolist())):].strip()
    return result.split("\n\n")[0].strip() or generated.strip()


def milestone_6(name: str, features: str, tone: str = "friendly", length: str = "medium") -> str:
    prompt = f"Product: {name}\nFeatures: {features}\nTone: {tone}\nDescription:"
    max_tokens = {"short": 90, "medium": 160, "long": 220}.get(length, 160)
    return milestone_5(prompt, max_new_tokens=max_tokens)
