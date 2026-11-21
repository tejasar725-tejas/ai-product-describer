"""Tests for all six project milestones."""
import torch

from app.core import (BPETokenizer, CausalSelfAttention, GPTConfig, GPTLanguageModel,
                      milestone_1, milestone_2, milestone_3, milestone_4,
                      milestone_5, milestone_6)


def test_milestone_1_collects_category_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.DATA_DIR", tmp_path)
    texts = milestone_1()
    assert len(texts) >= 5
    assert all(len(text) > 30 for text in texts)
    assert (tmp_path / "product_descriptions.txt").exists()


def test_milestone_2_bpe_round_trip_and_embeddings(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.MODEL_DIR", tmp_path)
    texts = ["cotton canvas tote", "cotton market bag", "canvas storage pouch"]
    tokenizer = milestone_2(texts, num_merges=20)
    sample = "cotton canvas tote"
    assert tokenizer.decode(tokenizer.encode(sample)) == sample
    embedding = torch.nn.Embedding(tokenizer.vocab_size, 16)
    assert embedding(torch.tensor(tokenizer.encode(sample))).shape[1] == 16
    assert (tmp_path / "tokenizer.json").exists()


def test_milestone_3_causal_attention_and_transformer_shapes():
    attention = CausalSelfAttention(16, 4, 8, dropout=0.0)
    x = torch.randn(2, 6, 16)
    assert attention(x).shape == x.shape
    model = milestone_3(vocab_size=300, block_size=8)
    logits, loss = model(torch.randint(0, 300, (2, 6)), torch.randint(0, 300, (2, 6)))
    assert logits.shape == (2, 6, 300)
    assert loss is not None and torch.isfinite(loss)


def test_milestone_4_trains_and_saves_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr("app.core.OUTPUT_DIR", tmp_path / "outputs")
    report = milestone_4(["A cotton tote with strong handles.", "A canvas pouch with a smooth zip."] * 3,
                         epochs=2, num_merges=10)
    assert report["epochs"] == 2
    assert (tmp_path / "models" / "product_gpt.pt").exists()
    assert (tmp_path / "outputs" / "training_metrics.json").exists()


def test_milestone_5_generates_text(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr("app.core.OUTPUT_DIR", tmp_path / "outputs")
    milestone_4(["A cotton tote with strong handles.", "A canvas pouch with a smooth zip."] * 3,
                epochs=2, num_merges=10)
    output = milestone_5("Product: Cotton tote. Description:", max_new_tokens=8)
    assert isinstance(output, str)


def test_milestone_6_cli_accepts_product_details(capsys):
    from app.cli import main
    main(["hello"])
    assert "ready" in capsys.readouterr().out
    import app.cli as cli
    cli.milestone_6 = lambda name, features, tone, length: f"{name}: {features}"
    main(["describe", "--name", "Tote", "--features", "cotton handles"])
    assert "Tote: cotton handles" in capsys.readouterr().out
