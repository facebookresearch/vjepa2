#!/usr/bin/env python3
import os, json
import numpy as np
from PIL import Image

def _device():
    import torch
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def _to_device(x, dev=None):
    import torch
    if dev is None: dev = _device()
    # tensors: allow non_blocking
    if isinstance(x, torch.Tensor):
        return x.to(dev, non_blocking=True)
    # modules & others: no non_blocking kw
    try:
        return x.to(dev)
    except Exception:
        try:
            return x.cuda() if dev.type == "cuda" else x.cpu()
        except Exception:
            return x

def _pick_module(obj):
    import torch.nn as nn
    if isinstance(obj, nn.Module): return obj
    if isinstance(obj, (list, tuple)):
        for it in obj:
            if isinstance(it, nn.Module):
                return it
    return obj  # fallback (some hubs return a callable wrapper)

def _latent_from_output(out):
    import torch
    # unwrap dict / tuple to a tensor-like
    if isinstance(out, dict):
        for k in ["features", "latent", "x", "emb", "last_hidden_state", "out"]:
            v = out.get(k, None)
            if isinstance(v, torch.Tensor):
                out = v; break
        else:
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    out = v; break
    elif isinstance(out, (list, tuple)):
        for v in out:
            if hasattr(v, "shape"):
                out = v; break

    if not hasattr(out, "detach"):
        raise RuntimeError(f"Cannot extract latent from type {type(out)}")

    z = out.detach().float().cpu().numpy()

    # strip batch if present
    if z.ndim >= 4 and z.shape[0] == 1:
        z = z[0]

    # [C,H,W] -> [H,W,C]
    if z.ndim == 3 and z.shape[0] in (256, 1536, 1408, 1024, 768) and z.shape[1] in (16,32) and z.shape[2] in (16,32):
        z = np.moveaxis(z, 0, -1)

    # token grid fallback: [T, D] -> [side, side, D]
    if z.ndim == 2:
        T, D = z.shape
        side = int(round(T**0.5))
        if side * side == T:
            z = z.reshape(side, side, D)
        else:
            z = np.broadcast_to(z[0], (16, 16, D)).copy()

    return z.astype(np.float32)

def _load_hub(model_key="vjepa2_vit_large"):
    import torch
    torch.set_grad_enabled(False)
    pre = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')
    raw = torch.hub.load('facebookresearch/vjepa2', model_key)
    enc = _pick_module(raw)
    enc.eval()
    enc = _to_device(enc)
    return pre, enc

def _load_hf(repo="facebook/vjepa2-vitl-fpc64-256"):
    import torch
    from transformers import AutoVideoProcessor, AutoModel
    torch.set_grad_enabled(False)
    dev = _device()
    pre = AutoVideoProcessor.from_pretrained(repo)
    enc = AutoModel.from_pretrained(repo)
    enc.eval()
    enc = _to_device(enc, dev)

    def hf_pre(img_pil):
        # returns torch tensor on proper device
        px = pre(images=img_pil, return_tensors="pt")["pixel_values"]  # [1,C,H,W]
        return _to_device(px, dev)
    return hf_pre, enc

def _encode_img(pre, enc, path, T_frames: int = 8):
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB")
    clip = [img] * T_frames  # V-JEPA2 preprocessor expects a clip (sequence)

    x = pre(clip)
    if isinstance(x, (list, tuple)):
        x = x[0]

    # Normalize to [B, C, T, H, W]
    if isinstance(x, torch.Tensor):
        if x.ndim == 5:
            # already [B, C, T, H, W]
            pass
        elif x.ndim == 4:
            # Heuristics:
            # - [C, T, H, W]  → add batch dim
            # - [T, C, H, W]  → permute to [C, T, H, W], then add batch dim
            # - [C, H, W]     → add T dimension and batch
            Cands = x.shape
            if Cands[0] in (8, 16, 32) and Cands[1] in (1, 3):
                # [T, C, H, W]
                x = x.permute(1, 0, 2, 3).unsqueeze(0)  # → [B, C, T, H, W]
            elif Cands[1] in (8, 16, 32):
                # [C, T, H, W]
                x = x.unsqueeze(0)  # → [B, C, T, H, W]
            elif x.shape[0] in (1, 3) and x.ndim == 4 and x.shape[2] > 32 and x.shape[3] > 32:
                # Some preprocessors give [C, H, W, T]
                # If the last dim is T-like, permute to [C, T, H, W]
                if x.shape[-1] in (8, 16, 32):
                    x = x.permute(0, 3, 1, 2).unsqueeze(0)  # [B, C, T, H, W]
                else:
                    # Treat as [C, H, W], add T
                    x = x.unsqueeze(0).unsqueeze(2).repeat(1, 1, T_frames, 1, 1)
            else:
                # Treat as [C, H, W], add T
                x = x.unsqueeze(0).unsqueeze(2).repeat(1, 1, T_frames, 1, 1)
        elif x.ndim == 3 and x.shape[0] in (1, 3):
            # [C, H, W] → add batch & T
            x = x.unsqueeze(0).unsqueeze(2).repeat(1, 1, T_frames, 1, 1)
        else:
            raise RuntimeError(f"Unexpected preprocessor tensor shape: {tuple(x.shape)}")
    else:
        raise RuntimeError(f"Preprocessor returned a non-tensor: {type(x)}")

    if torch.cuda.is_available():
        x = x.cuda()

    with torch.no_grad():
        z = enc(x)
        if isinstance(z, (list, tuple)):
            z = z[0]
        z = z.float().cpu().numpy()

    return z.astype("float32")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='data/start.png')
    ap.add_argument('--goal',  default='data/goal.png')
    ap.add_argument('--out_dir', default='siml/tmp')
    ap.add_argument('--backend', choices=['hub','hf'], default='hub')
    ap.add_argument('--hub_key', default='vjepa2_vit_large')  # smaller than vit_g
    ap.add_argument('--hf_repo', default='facebook/vjepa2-vitl-fpc64-256')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.backend == 'hub':
        pre, enc = _load_hub(args.hub_key)
    else:
        pre, enc = _load_hf(args.hf_repo)

    z_k = _encode_img(pre, enc, args.start)
    z_g = _encode_img(pre, enc, args.goal)

    np.save(os.path.join(args.out_dir, 'z_k.npy'), z_k)
    np.save(os.path.join(args.out_dir, 'z_g.npy'), z_g)
    meta = {
        "backend": args.backend,
        "hub_key": args.hub_key,
        "hf_repo": args.hf_repo,
        "z_k_shape": list(z_k.shape),
        "z_g_shape": list(z_g.shape),
    }
    with open(os.path.join(args.out_dir, 'latents_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("wrote:", os.path.join(args.out_dir,'z_k.npy'), os.path.join(args.out_dir,'z_g.npy'))
    print("latent shapes:", z_k.shape, z_g.shape)
    print("meta:", json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
