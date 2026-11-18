import json, math, pathlib, random, contextlib
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = {
    "qwen": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "smol": ("HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951"),
}
MAX_NEW = 64
SYS_SV = "Svara alltid på svenska."
import os
RES = pathlib.Path(os.environ.get("Z_RESULTS", ROOT / "results"))
SMOKE = int(os.environ.get("Z_SMOKE_N", "0"))

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.set_num_threads(4)

def load_model(key):
    name, rev = MODELS[key]
    tok = AutoTokenizer.from_pretrained(name, revision=rev, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(name, revision=rev, torch_dtype=torch.float32).eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok, model

def splits():
    sp = json.loads((ROOT / "data" / "splits.json").read_text(encoding="utf-8"))
    if SMOKE: sp = {k: v[:SMOKE] for k, v in sp.items()}
    return sp

def layers(model): return model.model.layers

def layer_grid(model):
    n = len(layers(model))
    return sorted({int(round(n * f)) for f in (0.25, 0.375, 0.5, 0.625, 0.75)})

# ---------- language id / spelling ----------
import langid
_lid = langid.classify
def lang_of(text): return _lid(text)[0]

_dic = None
def hunspell():
    global _dic
    if _dic is None:
        from spylls.hunspell import Dictionary
        _dic = Dictionary.from_files(str(ROOT / "data" / "hunspell" / "sv_SE"))
    return _dic

import re
_tokre = re.compile(r"[A-Za-zÅÄÖåäöéèüÉ]+(?:-[A-Za-zÅÄÖåäöéèüÉ]+)*")
def nonword_rate(text):
    """share of alphabetic tokens (>=2 chars) not accepted by sv_SE hunspell; truncated last token ignored"""
    toks = [t for t in _tokre.findall(text) if len(t) >= 2]
    if len(toks) > 1: toks = toks[:-1]  # last token may be cut by max_new_tokens
    if not toks: return None, 0
    d = hunspell()
    bad = sum(1 for t in toks if not (d.lookup(t) or d.lookup(t.lower()) or d.lookup(t.capitalize())))
    return bad / len(toks), len(toks)

# ---------- generation ----------
def build_prompts(tok, prompts, system=None):
    out = []
    for p in prompts:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}]
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out

@torch.no_grad()
def generate(tok, model, prompts, system=None, sample=False, seed=0, bs=32, max_new=None):
    texts = build_prompts(tok, prompts, system)
    outs, hit = [], []
    for i in range(0, len(texts), bs):
        seed_all(seed + i)
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True)
        kw = dict(do_sample=True, temperature=0.7, top_p=1.0, top_k=0) if sample else dict(do_sample=False)
        g = model.generate(**enc, max_new_tokens=max_new or MAX_NEW, pad_token_id=tok.pad_token_id, **kw)
        g = g[:, enc["input_ids"].shape[1]:]
        for row in g:
            ids = row.tolist()
            ended = tok.eos_token_id in ids or tok.convert_tokens_to_ids("<|im_end|>") in ids
            hit.append(not ended)
            outs.append(tok.decode(row, skip_special_tokens=True).strip())
    return outs, hit

# ---------- direction, edit, steering ----------
@torch.no_grad()
def mean_resid(tok, model, sents, L):
    """mean over sentences of token-mean-pooled residual stream at output of decoder layer L"""
    vecs = []
    for i in range(0, len(sents), 16):
        enc = tok(sents[i:i + 16], return_tensors="pt", padding=True)
        hs = model(**enc, output_hidden_states=True).hidden_states[L + 1]  # output of layer L
        m = enc["attention_mask"].unsqueeze(-1).float()
        vecs.append(((hs * m).sum(1) / m.sum(1)))
    return torch.cat(vecs).mean(0)

def direction(tok, model, en, sv, L):
    raw = mean_resid(tok, model, en, L) - mean_resid(tok, model, sv, L)
    return raw / raw.norm(), raw.norm().item()

@contextlib.contextmanager
def weight_edit(model, L, d, alpha):
    W = layers(model)[L].mlp.down_proj.weight  # (hidden, inter); output = W x
    orig = W.data.clone()
    W.data -= alpha * torch.outer(d, d @ orig)
    try: yield
    finally: W.data.copy_(orig)

def edit_stats(model, L):
    W = layers(model)[L].mlp.down_proj.weight
    total = sum(p.numel() for p in model.parameters())
    return {"layer": L, "matrix_entries_changed": W.numel(), "share_of_all_params": W.numel() / total,
            "free_parameters_in_edit": W.shape[0], "total_params": total}

@contextlib.contextmanager
def steering(model, L, vec):
    def hook(mod, inp, out):
        if isinstance(out, tuple): return (out[0] - vec.to(out[0].dtype),) + out[1:]
        return out - vec.to(out.dtype)
    h = layers(model)[L].register_forward_hook(hook)
    try: yield
    finally: h.remove()

def random_unit(dim, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()

# ---------- metrics ----------
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)

def mcnemar(a, b):
    """exact two-sided McNemar on paired boolean lists; returns (b01, b10, p)"""
    from scipy.stats import binomtest
    n01 = sum(1 for x, y in zip(a, b) if (not x) and y)
    n10 = sum(1 for x, y in zip(a, b) if x and (not y))
    n = n01 + n10
    p = 1.0 if n == 0 else binomtest(min(n01, n10), n, 0.5).pvalue
    return n01, n10, p

@torch.no_grad()
def perplexity(tok, model, texts, max_tok=96, bs=8):
    """token-level perplexity over the first max_tok tokens of each paragraph (right padded, pads masked)"""
    tok.padding_side = "right"
    nll, cnt = 0.0, 0
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_tok)
        logits = model(**enc).logits[:, :-1].float()
        tgt = enc["input_ids"][:, 1:]; m = enc["attention_mask"][:, 1:].bool()
        lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        nll += -(lp[m]).sum().item(); cnt += m.sum().item()
    tok.padding_side = "left"
    return math.exp(nll / cnt)
