"""Byte-level BPE and a compact GPT-style decoder-only Transformer."""
from collections import Counter
from pathlib import Path
import json, math, random
import torch
from torch import nn
from torch.nn import functional as F

class BPETokenizer:
    def __init__(self):
        self.merges=[]; self.token_to_id={bytes([i]):i for i in range(256)}; self.id_to_token={i:bytes([i]) for i in range(256)}
    @property
    def vocab_size(self): return len(self.token_to_id)
    def train(self,texts,vocab_size=320):
        if vocab_size<256: raise ValueError('vocab_size must be at least 256')
        corpus=[list(t.encode('utf-8')) for t in texts if t]
        self.__init__()
        while len(self.token_to_id)<vocab_size:
            counts=Counter(pair for seq in corpus for pair in zip(seq,seq[1:]))
            if not counts: break
            pair,n=min(counts.items(),key=lambda x:(-x[1],x[0]))
            if n<2: break
            a,b=pair; merged=self.id_to_token[a]+self.id_to_token[b]; new=len(self.token_to_id)
            self.merges.append((self.id_to_token[a],self.id_to_token[b])); self.token_to_id[merged]=new; self.id_to_token[new]=merged
            updated=[]
            for seq in corpus:
                out=[]; i=0
                while i<len(seq):
                    if i+1<len(seq) and (seq[i],seq[i+1])==pair: out.append(new); i+=2
                    else: out.append(seq[i]); i+=1
                updated.append(out)
            corpus=updated
        return self
    def encode(self,text):
        seq=list(text.encode('utf-8'))
        for i,(a,b) in enumerate(self.merges):
            x,y=self.token_to_id[a],self.token_to_id[b]; new=256+i; out=[]; j=0
            while j<len(seq):
                if j+1<len(seq) and seq[j]==x and seq[j+1]==y: out.append(new); j+=2
                else: out.append(seq[j]); j+=1
            seq=out
        return seq
    def decode(self,ids): return b''.join(self.id_to_token[int(i)] for i in ids).decode('utf-8',errors='replace')
    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps([[a.hex(),b.hex()] for a,b in self.merges]),encoding='utf-8')
    @classmethod
    def load(cls,path):
        obj=cls()
        for ah,bh in json.loads(Path(path).read_text(encoding='utf-8')):
            a,b=bytes.fromhex(ah),bytes.fromhex(bh); idx=len(obj.token_to_id); obj.merges.append((a,b)); obj.token_to_id[a+b]=idx; obj.id_to_token[idx]=a+b
        return obj

class CausalSelfAttention(nn.Module):
    def __init__(self,n_embd,n_head,block_size,dropout=.1):
        super().__init__()
        if n_embd%n_head: raise ValueError('n_embd must be divisible by n_head')
        self.n_head=n_head; self.head_dim=n_embd//n_head; self.qkv=nn.Linear(n_embd,3*n_embd); self.proj=nn.Linear(n_embd,n_embd); self.drop=nn.Dropout(dropout)
        self.register_buffer('mask',torch.tril(torch.ones(block_size,block_size,dtype=torch.bool)))
    def forward(self,x):
        b,t,c=x.shape; q,k,v=self.qkv(x).chunk(3,dim=-1)
        reshape=lambda z:z.view(b,t,self.n_head,self.head_dim).transpose(1,2)
        q,k,v=map(reshape,(q,k,v)); scores=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim); scores=scores.masked_fill(~self.mask[:t,:t],float('-inf'))
        y=self.drop(F.softmax(scores,dim=-1))@v; y=y.transpose(1,2).contiguous().view(b,t,c)
        return self.drop(self.proj(y))

class TransformerBlock(nn.Module):
    def __init__(self,n_embd,n_head,block_size,dropout):
        super().__init__(); self.ln1=nn.LayerNorm(n_embd); self.attn=CausalSelfAttention(n_embd,n_head,block_size,dropout); self.ln2=nn.LayerNorm(n_embd)
        self.ff=nn.Sequential(nn.Linear(n_embd,4*n_embd),nn.GELU(),nn.Linear(4*n_embd,n_embd),nn.Dropout(dropout))
    def forward(self,x):
        x=x+self.attn(self.ln1(x)); return x+self.ff(self.ln2(x))

class GPTLanguageModel(nn.Module):
    def __init__(self,vocab_size,block_size=96,n_embd=64,n_head=4,n_layer=2,dropout=.1):
        super().__init__(); self.vocab_size=vocab_size; self.block_size=block_size; self.token_embedding=nn.Embedding(vocab_size,n_embd); self.position_embedding=nn.Embedding(block_size,n_embd)
        self.blocks=nn.Sequential(*[TransformerBlock(n_embd,n_head,block_size,dropout) for _ in range(n_layer)]); self.norm=nn.LayerNorm(n_embd); self.head=nn.Linear(n_embd,vocab_size)
    def forward(self,idx,targets=None):
        b,t=idx.shape
        if t>self.block_size: raise ValueError('input exceeds block_size')
        pos=torch.arange(t,device=idx.device); x=self.token_embedding(idx)+self.position_embedding(pos); logits=self.head(self.norm(self.blocks(x)))
        loss=F.cross_entropy(logits.reshape(-1,self.vocab_size),targets.reshape(-1)) if targets is not None else None
        return logits,loss
    @torch.no_grad()
    def generate(self,idx,max_new_tokens=80,temperature=.9,top_k=40):
        if temperature<=0: raise ValueError('temperature must be positive')
        self.eval()
        for _ in range(max_new_tokens):
            logits,_=self(idx[:,-self.block_size:]); logits=logits[:,-1,:]/temperature
            if top_k>0:
                vals,_=torch.topk(logits,min(top_k,logits.size(-1))); logits[logits<vals[:,-1,None]]=float('-inf')
            idx=torch.cat((idx,torch.multinomial(F.softmax(logits,dim=-1),1)),dim=1)
        return idx

def train_model(texts,epochs=60,batch_size=16,block_size=96,device='cpu',seed=42):
    if not texts: raise ValueError('training corpus is empty')
    if epochs<1 or batch_size<1 or block_size<2: raise ValueError('invalid training parameters')
    random.seed(seed); torch.manual_seed(seed); tokenizer=BPETokenizer().train(texts,320); ids=tokenizer.encode('\n\n'.join(texts)+'\n\n')
    data=torch.tensor(ids,dtype=torch.long)
    if len(data)<=block_size: data=data.repeat(block_size//max(1,len(data))+2)
    model=GPTLanguageModel(tokenizer.vocab_size,block_size=block_size).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4); losses=[]
    for _ in range(epochs):
        starts=torch.randint(0,len(data)-block_size-1,(batch_size,)); x=torch.stack([data[int(i):int(i)+block_size] for i in starts]).to(device); y=torch.stack([data[int(i)+1:int(i)+block_size+1] for i in starts]).to(device)
        model.train(); _,loss=model(x,y); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach().cpu()))
    return model,tokenizer,losses
