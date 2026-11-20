import torch
def generate_description(model,tokenizer,prompt,max_new_tokens=80,temperature=.9,top_k=40,device='cpu'):
    ids=tokenizer.encode(prompt) or [32]; x=torch.tensor([ids],dtype=torch.long,device=device)
    return tokenizer.decode(model.generate(x,max_new_tokens,temperature,top_k)[0].tolist())
