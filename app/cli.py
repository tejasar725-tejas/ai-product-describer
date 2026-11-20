import argparse
from pathlib import Path
import torch
from .core import train_model
from .storage import load_corpus,save_checkpoint,load_checkpoint
from .llm import generate_description

def build_parser():
    p=argparse.ArgumentParser(description='Train a small GPT and generate product descriptions.'); sub=p.add_subparsers(dest='command')
    t=sub.add_parser('train'); t.add_argument('--corpus'); t.add_argument('--epochs',type=int,default=60); t.add_argument('--batch-size',type=int,default=16); t.add_argument('--block-size',type=int,default=96); t.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); t.add_argument('--model-path',default='artifacts/model.pt'); t.add_argument('--tokenizer-path',default='artifacts/tokenizer.json')
    g=sub.add_parser('generate'); g.add_argument('--product',required=True); g.add_argument('--features',default=''); g.add_argument('--tone',choices=['friendly','elegant','playful','professional','minimal'],default='friendly'); g.add_argument('--max-new-tokens',type=int,default=80); g.add_argument('--temperature',type=float,default=.9); g.add_argument('--top-k',type=int,default=40); g.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); g.add_argument('--model-path',default='artifacts/model.pt'); g.add_argument('--tokenizer-path',default='artifacts/tokenizer.json'); g.add_argument('--output')
    return p

def main(argv=None):
    p=build_parser(); a=p.parse_args(argv)
    if a.command=='train':
        if a.epochs<1 or a.batch_size<1 or a.block_size<2: p.error('epochs/batch-size must be positive and block-size >= 2')
        texts=load_corpus(a.corpus); model,tok,losses=train_model(texts,a.epochs,a.batch_size,a.block_size,a.device); save_checkpoint(model,tok,a.model_path,a.tokenizer_path)
        print(f'Trained on {len(texts)} descriptions for {a.epochs} epochs. Final loss: {losses[-1]:.4f}'); print(f'Saved {a.model_path} and {a.tokenizer_path}'); return 0
    if a.command=='generate':
        if a.max_new_tokens<1 or a.temperature<=0: p.error('generation length and temperature must be positive')
        model,tok=load_checkpoint(a.model_path,a.tokenizer_path,a.device); prompt=f'Product: {a.product}. Features: {a.features}. Tone: {a.tone}. Description:'; result=generate_description(model,tok,prompt,a.max_new_tokens,a.temperature,a.top_k,a.device); print('\n'+result)
        if a.output: Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(result+'\n',encoding='utf-8'); print(f'Saved to {a.output}')
        return 0
    p.print_help(); return 0
if __name__=='__main__': raise SystemExit(main())
