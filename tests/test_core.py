import torch
from app.core import BPETokenizer,CausalSelfAttention,GPTLanguageModel

def test_bpe_utf8_round_trip():
    t=BPETokenizer().train(['fresh rosewater mist','gentle oat cleanser'],280); s='Rosewater mist – café'; assert t.decode(t.encode(s))==s

def test_bpe_save_load(tmp_path):
    t=BPETokenizer().train(['fresh cream','fresh lotion'],270); p=tmp_path/'tok.json'; t.save(p); u=BPETokenizer.load(p); assert u.decode(u.encode('fresh cream'))=='fresh cream'

def test_attention_shape():
    a=CausalSelfAttention(16,4,12,0); assert a(torch.randn(2,7,16)).shape==(2,7,16)

def test_gpt_forward_loss():
    m=GPTLanguageModel(270,12,16,4,1,0); x=torch.randint(0,270,(2,8)); logits,loss=m(x,x); assert logits.shape==(2,8,270) and torch.isfinite(loss)
