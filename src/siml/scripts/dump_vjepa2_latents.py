import torch, numpy as np
from PIL import Image

# PyTorch Hub paths per their README, or use HF AutoModel/AutoVideoProcessor
enc = torch.hub.load('facebookresearch/vjepa2','vjepa2_vit_giant').eval().cuda()
pre = torch.hub.load('facebookresearch/vjepa2','vjepa2_preprocessor')

def encode_img(path):
    img = Image.open(path).convert('RGB')
    x = pre(img)  # their preprocessor to 256p
    with torch.no_grad():
        z = enc.encode(x.to('cuda'))  # adapt to their API; returns (16,16,1408)
    return z.detach().cpu().numpy()

z_k = encode_img('data/start.png')
z_g = encode_img('data/goal.png')

np.save('siml/tmp/z_k.npy', z_k)
np.save('siml/tmp/z_g.npy', z_g)
print('wrote siml/tmp/z_k.npy & z_g.npy')