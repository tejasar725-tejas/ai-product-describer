from pathlib import Path
import torch
DEFAULT_CORPUS=Path(__file__).resolve().parent.parent/'data'/'product_descriptions.txt'
def load_corpus(path=None):
    p=Path(path) if path else DEFAULT_CORPUS
    if not p.is_file(): raise FileNotFoundError(f'Corpus not found: {p}')
    return [s.strip() for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def save_checkpoint(model,tokenizer,model_path,tokenizer_path):
    Path(model_path).parent.mkdir(parents=True,exist_ok=True); Path(tokenizer_path).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state':model.state_dict(),'config':{'vocab_size':model.vocab_size,'block_size':model.block_size,'n_embd':model.token_embedding.embedding_dim,'n_head':model.blocks[0].attn.n_head,'n_layer':len(model.blocks)}},model_path); tokenizer.save(tokenizer_path)
def load_checkpoint(model_path,tokenizer_path,device='cpu'):
    from .core import GPTLanguageModel,BPETokenizer
    ck=torch.load(model_path,map_location=device,weights_only=True); model=GPTLanguageModel(**ck['config']).to(device); model.load_state_dict(ck['state']); model.eval(); return model,BPETokenizer.load(tokenizer_path)
