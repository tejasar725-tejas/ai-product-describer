# AI Product Describer

An educational command-line tool that trains a small GPT-style decoder-only Transformer from scratch on skincare product descriptions and generates new product copy.

## Features
- Sample skincare product-description corpus (one example per line)
- Byte-level Byte-Pair Encoding tokenizer trained on the corpus
- Learned token and position embeddings
- Multi-head causal self-attention with a lower-triangular mask
- Transformer decoder blocks, feed-forward layers and residual connections
- Next-token training with cross-entropy loss and AdamW
- Temperature/top-k sampling
- CLI training, generation, tone selection and saving output to a file

## Windows PowerShell setup
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train
```powershell
python -m app.cli train --epochs 60
```
Creates `artifacts/model.pt` and `artifacts/tokenizer.json`.

## Generate
```powershell
python -m app.cli generate --product "Rosewater face mist" --features "fine mist, refreshing, travel friendly" --tone elegant --max-new-tokens 70
```
Save a description:
```powershell
python -m app.cli generate --product "Oat cleanser" --features "gentle, fragrance free" --output outputs/oat-cleanser.txt
```

## Tests
```powershell
python -m pytest -v
```

## Project structure
```text
ai-product-describer/
  .env.example
  .github/workflows/ci.yml
  .gitignore
  README.md
  app/__init__.py
  app/cli.py
  app/core.py
  app/llm.py
  app/storage.py
  data/product_descriptions.txt
  pytest.ini
  requirements.txt
  tests/test_core.py
  tests/test_smoke.py
```

This is a tiny educational GPT, not a production-grade foundation model. The starter corpus is synthetic and intentionally small; output quality will be limited. Replace it with a larger, permission-cleared dataset for stronger results. Training from scratch can take several minutes on CPU.
