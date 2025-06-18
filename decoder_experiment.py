import math

import torch
from transformers import AutoModel, AutoVideoProcessor

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from models.decoder import VJEPA2Decoder

hf_repo = "facebook/vjepa2-vitl-fpc64-256"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained(hf_repo).to(device)
processor = AutoVideoProcessor.from_pretrained(hf_repo).to(device)
decoder = VJEPA2Decoder(model.config, model.encoder.embeddings).to(device)

dataset = LeRobotDataset("lerobot/droid_100")

optimizer = torch.optim.AdamW(decoder.parameters(), lr=0, weight_decay=0.1)
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=8,
    shuffle=True,
    pin_memory=device.type != "cpu",
    drop_last=True,
)


def lr_lambda(current_step: int):
    if current_step < 2000:
        # Linear warmup
        return float(current_step) / 2000.0
    else:
        # Cosine decay
        progress = float(current_step - 2000) / (150000 - 2000)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

decoder.train()
num_steps = 10
for step in range(num_steps):
    for batch in dataloader:
        image1 = processor(batch['observation.images.exterior_image_1_left'], return_tensors="pt").to(model.device)[
            "pixel_values_videos"]
        image2 = processor(batch['observation.images.exterior_image_2_left'], return_tensors="pt").to(model.device)[
            "pixel_values_videos"]

        pixel_values = torch.cat([image1, image2], dim=1)
        outputs = model(pixel_values, skip_predictor=True)
        reconstructed = decoder(outputs.last_hidden_state)

        loss = torch.nn.functional.mse_loss(reconstructed, pixel_values)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()

        print(f"Step {step + 1}/{num_steps}, Loss: {loss.item()}")

    scheduler.step()
    torch.save(decoder.state_dict(), f"decoder_{step}.pt")


torch.save(decoder.state_dict(), "decoder_final.pt")
