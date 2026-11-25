"""Model-side measurements for Semantic Gravity v2 (DESIGN_v2.md, "Measurements per prompt").

Everything here works for any Qwen2-family causal LM; layer count, hidden size and vocab
size are read from the loaded model. Scoring passes (P0/P1, logit lens, attention, DLA,
patching) use RIGHT padding and read each row at its own decision position, so padding
never touches the positions we read. Generation uses LEFT padding, as the design requires.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

BASELINE_TEMPLATE = "Answer with exactly one English word.\nQuestion: {question}\nAnswer:"
NEGATIVE_TEMPLATE = (
    "Answer with exactly one English word.\n"
    'Do not use the word "{X}" anywhere in your answer.\n'
    "Question: {question}\nAnswer:"
)
CONDITIONS = ("base", "neg")
P0_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def p0_bin(p: float) -> int:
    """Bins [0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1]."""
    return int(min(max(int(p * 5), 0), 4))


# ----------------------------------------------------------------------------- text rules

_WORD_CHAR = r"[^\W_]"  # a Unicode letter or digit, same rule as str.isalnum()


def contains_word(target: str, text: str) -> bool:
    """X appears as a whole word: case-insensitive, bounded by non-letter/digit or text edge."""
    pat = rf"(?<!{_WORD_CHAR}){re.escape(target)}(?!{_WORD_CHAR})"
    return re.search(pat, text, flags=re.IGNORECASE) is not None


def surface_variants(X: str) -> list[str]:
    """{X, x, X.capitalize(), X.upper()} deduplicated, each without and with a leading space."""
    forms: list[str] = []
    for f in (X, X.lower(), X.capitalize(), X.upper()):
        if f not in forms:
            forms.append(f)
    return forms + [" " + f for f in forms]


def user_message(question: str, X: str, cond: str) -> str:
    if cond == "base":
        return BASELINE_TEMPLATE.format(question=question)
    return NEGATIVE_TEMPLATE.format(question=question, X=X)


# ----------------------------------------------------------------------------- model

@dataclass
class LM:
    model: torch.nn.Module
    tok: object
    device: torch.device
    n_layers: int
    starts_alnum: torch.Tensor  # [V] bool: decoded token starts with a letter or digit
    w_mean: torch.Tensor  # [d] fp32 mean unembedding row, for centred DLA
    eos_ids: list[int]
    pad_id: int


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_lm(model_path: str, device: str = "auto", dtype: str = "auto") -> LM:
    dev = pick_device(device)
    if dtype == "auto":
        dtype = "bfloat16" if dev.type == "cuda" else "float32"
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"  # generation; scoring builds its own right-padded tensors
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=getattr(torch, dtype), attn_implementation="sdpa"
    ).to(dev)
    model.eval()
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[1]
    # A clean generation config, so nothing from the model's hub defaults (Qwen ships
    # top_k=20, top_p=0.8, repetition_penalty=1.1) can leak into greedy or sampling.
    model.generation_config = GenerationConfig(eos_token_id=eos, pad_token_id=pad)
    V = model.lm_head.weight.shape[0]
    first_chars = []
    for i in range(V):
        s = tok.decode([i]) if i < len(tok) else ""
        first_chars.append(bool(s) and s[0].isalnum())
    starts_alnum = torch.tensor(first_chars, dtype=torch.bool, device=dev)
    with torch.no_grad():
        W = model.lm_head.weight
        w_mean = sum(c.float().sum(0) for c in W.split(8192)) / W.shape[0]
    return LM(model, tok, dev, model.config.num_hidden_layers, starts_alnum, w_mean, eos, pad)


def set_attn(lm: LM, impl: str) -> None:
    if lm.model.config._attn_implementation != impl:
        lm.model.set_attn_implementation(impl)


# ----------------------------------------------------------------------------- prompts

@dataclass
class Encoded:
    ids: list[int]
    text: str
    spans: dict  # name -> list of token indices (neg only: tmf, nf, q)
    span_text: dict  # name -> decoded text of the span


def _apply_template(tok, user: str, tokenize: bool):
    out = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=tokenize, add_generation_prompt=True
    )
    if tokenize and not isinstance(out, list):
        out = out["input_ids"]
    return out


def encode_prompt(tok, question: str, X: str, cond: str) -> Encoded:
    user = user_message(question, X, cond)
    text = _apply_template(tok, user, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    ref = list(_apply_template(tok, user, tokenize=True))
    assert ids == ref, "offset tokenization differs from chat-template tokenization"
    spans, span_text = {}, {}
    if cond == "neg":
        u0 = text.index(user)
        tmf_a = u0 + user.index(f'"{X}"') + 1
        nf_a = u0 + user.index("Do not")
        q_a = u0 + user.index("Question: ") + len("Question: ")
        char_spans = {"tmf": (tmf_a, tmf_a + len(X)), "nf": (nf_a, nf_a + 6),
                      "q": (q_a, q_a + len(question))}
        offs = enc["offset_mapping"]
        for name, (a, b) in char_spans.items():
            idx = [i for i, (s, e) in enumerate(offs) if s < b and e > a]
            spans[name] = idx
            span_text[name] = tok.decode([ids[i] for i in idx])
    return Encoded(ids, text, spans, span_text)


def span_ok(e: Encoded, X: str, question: str) -> bool:
    t = e.span_text
    return (t["tmf"].strip().strip('"').strip() == X and t["nf"].strip() == "Do not"
            and t["q"].strip() == question.strip())


# ----------------------------------------------------------------------------- batching

def right_pad(seqs: list[list[int]], pad: int, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


def left_pad(seqs: list[list[int]], pad: int, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s)
        mask[i, L - len(s):] = 1
    return ids.to(device), mask.to(device)


def _is_oom(e: Exception) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def run_split(fn, items: list, batch: int):
    """Apply fn to items in batches; on out-of-memory, halve the batch and retry."""
    out, i = [], 0
    while i < len(items):
        chunk = items[i: i + batch]
        try:
            out.extend(fn(chunk))
            i += len(chunk)
        except Exception as e:  # noqa: BLE001
            if not _is_oom(e) or batch == 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch = max(1, batch // 2)
            print(f"[oom] batch reduced to {batch}", flush=True)
    return out


# ----------------------------------------------------------------------------- measurement 1

@torch.no_grad()
def score_variants(lm: LM, prompt_ids: list[list[int]], targets: list[str], batch: int):
    """Exact P(first word is X) per prompt (measurement 1).

    For each variant v: P(canonical tokens of v) * P(next token does not start with a
    letter or digit | prompt, v). Returns per prompt (total, [per-variant P], variants,
    [variant token ids]).
    """
    tok = lm.tok
    jobs = []  # (prompt index, variant index, seq, n_prompt, var_ids)
    meta = []
    for pi, (ids, X) in enumerate(zip(prompt_ids, targets)):
        vs = surface_variants(X)
        vids = [tok.encode(v, add_special_tokens=False) for v in vs]
        meta.append((vs, vids))
        for vi, v_ids in enumerate(vids):
            jobs.append((pi, vi, ids + v_ids, len(ids), v_ids))

    def fn(chunk):
        ids, mask = right_pad([j[2] for j in chunk], lm.pad_id, lm.device)
        h = lm.model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        # Gather every position we need from the whole batch, one lm_head call.
        r_idx, p_idx, tgt, owner = [], [], [], []
        for r, (_, _, seq, n, v_ids) in enumerate(chunk):
            for j in range(len(v_ids) + 1):  # predicts v_ids[j]; the last predicts the end
                r_idx.append(r); p_idx.append(n - 1 + j)
                tgt.append(v_ids[j] if j < len(v_ids) else -1); owner.append(r)
        lp = torch.log_softmax(lm.model.lm_head(h[r_idx, p_idx]).float(), dim=-1)
        tgt_t = torch.tensor(tgt, device=lm.device)
        owner_t = torch.tensor(owner, device=lm.device)
        is_tok = tgt_t >= 0
        tok_lp = torch.zeros(len(chunk), device=lm.device).index_add_(
            0, owner_t[is_tok], lp[is_tok].gather(1, tgt_t[is_tok, None]).squeeze(1))
        end_p = lp[~is_tok].exp().masked_fill(lm.starts_alnum, 0.0).sum(-1)  # rows in order
        return (tok_lp.exp() * end_p).tolist()

    probs = run_split(fn, jobs, batch)
    per = [[0.0] * len(m[0]) for m in meta]
    for (pi, vi, *_), p in zip(jobs, probs):
        per[pi][vi] = p
    out = []
    for pi, (vs, vids) in enumerate(meta):
        total = float(sum(per[pi]))
        assert total <= 1 + 1e-4, f"P={total} > 1 for prompt {pi}"
        out.append((total, per[pi], vs, vids))
    return out


# ----------------------------------------------------------------------------- measurements 5-7

@torch.no_grad()
def hooked_forward(lm: LM, seqs: list[list[int]], want_attn: bool, patch=None):
    """One right-padded forward pass; captures, at each row's decision position (its last
    token) and for every layer: resid_post (decoder layer output, before the final norm),
    attention output, MLP output, and (if want_attn) attention weights of the decision
    query averaged over heads. Returns (dict of tensors, last-position logits in fp32).

    patch: optional dict with 'kind' (list per row: None/'R'/'A'), 'layer' (list per
    row), 'vec' ([rows, d] tensor). Row r gets its residual ('R') or attention output
    ('A') at layer 'layer[r]' replaced by vec[r] at its decision position.
    """
    m = lm.model.model
    ids, mask = right_pad(seqs, lm.pad_id, lm.device)
    B = len(seqs)
    pos = torch.tensor([len(s) - 1 for s in seqs], device=lm.device)
    rows = torch.arange(B, device=lm.device)
    cap = {"resid": [None] * lm.n_layers, "attn": [None] * lm.n_layers,
           "mlp": [None] * lm.n_layers, "attw": [None] * lm.n_layers}
    hooks = []
    if patch is not None:
        kinds, layers = patch["kind"], patch["layer"]
        vec = patch["vec"].to(lm.device)
        sel = {}
        for r, (k, l) in enumerate(zip(kinds, layers)):
            if k is not None:
                sel.setdefault((k, l), []).append(r)
        sel = {k: torch.tensor(v, device=lm.device) for k, v in sel.items()}

    def _replace(t, key):
        if patch is None or key not in sel:
            return t
        r = sel[key]
        t = t.clone()
        t[r, pos[r]] = vec[r].to(t.dtype)
        return t

    for l, layer in enumerate(m.layers[: lm.n_layers]):
        def attn_hook(mod, inp, out, l=l):
            a, w = out[0], out[1]
            a = _replace(a, ("A", l))
            cap["attn"][l] = a[rows, pos]
            if want_attn:
                cap["attw"][l] = w[rows, :, pos, :].float().mean(dim=1)  # [B, T]
            return (a, w) + tuple(out[2:])

        def mlp_hook(mod, inp, out, l=l):
            cap["mlp"][l] = out[rows, pos]

        def layer_hook(mod, inp, out, l=l):
            h = out[0] if isinstance(out, tuple) else out
            h = _replace(h, ("R", l))
            cap["resid"][l] = h[rows, pos]
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        hooks += [layer.self_attn.register_forward_hook(attn_hook),
                  layer.mlp.register_forward_hook(mlp_hook),
                  layer.register_forward_hook(layer_hook)]
    try:
        h = m(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    finally:
        for hk in hooks:
            hk.remove()
    logits = lm.model.lm_head(h[rows, pos]).float()
    out = {k: torch.stack(v, 1) for k, v in cap.items() if v[0] is not None}  # [B, L, ...]
    out["embed"] = m.embed_tokens(ids[rows, pos])
    return out, logits


@torch.no_grad()
def lens_dist(lm: LM, resid_l: torch.Tensor) -> torch.Tensor:
    """Logit-lens next-token distribution from one layer's resid_post [B, d]: the final
    norm and unembedding in the model's dtype (exactly the model's own last-layer path),
    softmax in fp32."""
    return torch.softmax(lm.model.lm_head(lm.model.model.norm(resid_l)).float(), -1)


@torch.no_grad()
def lens_probs(lm: LM, resid: torch.Tensor, tok_ids: list[list[int]]):
    """Logit lens: softmax(lm_head(final_norm(resid_post_l))) at every layer, computed in
    the model's dtype exactly as the model does at its last layer, softmax in fp32.
    tok_ids[b] lists the target-token set of row b, with the target token t* first.
    Returns [B, L] prob of t* and of the whole set."""
    B, L, _ = resid.shape
    K = max(len(t) for t in tok_ids)
    idx = torch.tensor([t + [t[0]] * (K - len(t)) for t in tok_ids], device=resid.device)
    valid = torch.tensor([[1.0] * len(t) + [0.0] * (K - len(t)) for t in tok_ids],
                         device=resid.device)
    p_tok, p_set = [], []
    for l in range(L):
        pr = lens_dist(lm, resid[:, l])
        g = pr.gather(1, idx)
        p_tok.append(g[:, 0]); p_set.append((g * valid).sum(1))
    return torch.stack(p_tok, 1).cpu(), torch.stack(p_set, 1).cpu()


@torch.no_grad()
def dla(lm: LM, cap: dict, t_star: torch.Tensor):
    """Linearised direct logit attribution onto W_U[t*] (raw) and onto W_U[t*] - mean_v W_U[v]
    (centred), using the final-norm scale of the real final residual at the decision
    position. Returns dict of [B, L] arrays plus embedding terms [B]."""
    norm = lm.model.model.norm
    w_raw = lm.model.lm_head.weight[t_star].float()  # [B, d]
    w_cen = w_raw - lm.w_mean
    final = cap["resid"][:, -1].float()
    sigma = torch.sqrt(final.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)  # [B,1]
    g = norm.weight.float()
    res = {}
    for name in ("attn", "mlp"):
        c = cap[name].float() * g / sigma[:, None]  # [B, L, d]
        res[f"dla_{name}"] = torch.einsum("bld,bd->bl", c, w_raw).cpu()
        res[f"dla_{name}_c"] = torch.einsum("bld,bd->bl", c, w_cen).cpu()
    e = cap["embed"].float() * g / sigma
    res["dla_embed"] = (e * w_raw).sum(-1).cpu()
    res["dla_embed_c"] = (e * w_cen).sum(-1).cpu()
    return res


def attention_masses(attw: torch.Tensor, spans: list[dict]):
    """attw [B, L, T] (head-mean attention of the decision query). Returns tmf, nf, q [B, L]."""
    out = {}
    for name in ("tmf", "nf", "q"):
        vals = torch.full(attw.shape[:2], float("nan"))
        for b, sp in enumerate(spans):
            if sp and sp.get(name):
                vals[b] = attw[b][:, sp[name]].sum(-1).cpu()
        out[f"attn_{name}"] = vals
    return out


# ----------------------------------------------------------------------------- measurements 2-3

@torch.no_grad()
def generate(lm: LM, prompt_ids: list[list[int]], *, do_sample: bool, n: int,
             max_new_tokens: int, temperature: float = 1.0, top_p: float = 1.0,
             seq_batch: int = 512) -> list[list[str]]:
    """Batched generation with LEFT padding. Returns n decoded completions per prompt."""
    assert lm.tok.padding_side == "left"
    cfg = GenerationConfig(
        do_sample=do_sample, temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None, top_k=0 if do_sample else None,
        repetition_penalty=1.0,
        max_new_tokens=max_new_tokens, num_return_sequences=n,
        eos_token_id=lm.eos_ids, pad_token_id=lm.pad_id,
    )

    def fn(chunk):
        ids, mask = left_pad(chunk, lm.pad_id, lm.device)
        out = lm.model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg)
        new = out[:, ids.shape[1]:]
        texts = lm.tok.batch_decode(new, skip_special_tokens=True)
        return [texts[i * n:(i + 1) * n] for i in range(len(chunk))]

    return run_split(fn, prompt_ids, max(1, seq_batch // n))


def to_np(t) -> np.ndarray:
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
