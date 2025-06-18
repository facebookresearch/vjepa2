from typing import Optional

import torch
import torch.nn as nn
from transformers import VJEPA2Config
from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Layer, VJEPA2PreTrainedModel, VJEPA2Model, \
    VJEPA2PatchEmbeddings3D


class InverseVJEPA2PatchEmbeddings3D(nn.Module):
    """
    Inverts the VJEPA2PatchEmbeddings3D operation.
    """

    def __init__(
            self,
            config: VJEPA2Config,
            hidden_size: int = 1024,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.tubelet_size = config.tubelet_size

        self.num_patches_per_dim = config.crop_size // config.patch_size

        self.inv_proj = nn.ConvTranspose3d(
            in_channels=hidden_size,
            out_channels=config.in_chans,
            kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
            stride=(config.tubelet_size, config.patch_size, config.patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)

        x = x.view(x.shape[0], self.hidden_size, 8, self.num_patches_per_dim, self.num_patches_per_dim)

        x = self.inv_proj(x)
        return x.transpose(1, 2)


class VJEPA2Decoder(nn.Module):
    """
    VJEPA2 Decoder - A feedforward network that reconstructs images from encoded representations.
    Parameterized as a ViT-L architecture with output dimension 256 × 256 × 3.
    """

    def __init__(self, config: VJEPA2Config, embeddings: VJEPA2PatchEmbeddings3D):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size

        # Define transformer layers for processing the encoded tokens
        drop_path_rates = [
            (config.drop_path_rate * i / (config.num_hidden_layers - 1) if config.num_hidden_layers > 1 else 0.0)
            for i in range(config.num_hidden_layers)
        ]
        self.layers = nn.ModuleList(
            [
                VJEPA2Layer(
                    config,
                    drop_path_rate=drop_path_rates[i],
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.inverse_patch_embed = InverseVJEPA2PatchEmbeddings3D(config)
        self._load_patch_embed_weights(embeddings)

        # Final layer normalization
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.decoder_pred = nn.Linear(self.hidden_size, )

    def _load_patch_embed_weights(self, embeddings: VJEPA2PatchEmbeddings3D):
        with torch.no_grad():
            self.inverse_patch_embed.inv_proj.weight.data = embeddings.patch_embeddings.proj.weight.data

    # def unpatchify(self, x):
    #     p = self.patch_size
    #
    #     # Calculate grid size (number of patches in each dimension)
    #     grid_size = int(math.sqrt(x.shape[1] + 0.001))  # Add small epsilon to handle floating point errors
    #
    #     # First reshape to separate batch, grid dimensions, and patch content
    #     x = x.reshape(x.shape[0], grid_size, grid_size, p * p * 3)
    #
    #     # Then reshape to separate the patch dimensions
    #     x = x.reshape(x.shape[0], grid_size, grid_size, p, p, 3)
    #
    #     # Permute and reshape to image format (B, 3, H, W)
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], 3, grid_size * p, grid_size * p))
    #
    #     return imgs

    def forward(self, encoded_states, head_mask=None):
        hidden_states = encoded_states

        # Process through transformer layers
        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, None, layer_head_mask, False)
            hidden_states = layer_outputs[0]

        # Apply final normalization
        hidden_states = self.layernorm(hidden_states)

        # Project to patch pixel values
        # x = self.decoder_pred(hidden_states)

        # Reshape patches to image
        # reconstructed_pixel_values = self.unpatchify(x)
        reconstructed_pixel_values = self.inverse_patch_embed(hidden_states)

        return reconstructed_pixel_values


class VJEPA2ForImageReconstruction(VJEPA2PreTrainedModel):
    def __init__(self, config: VJEPA2Config, embeddings: VJEPA2PatchEmbeddings3D):
        super().__init__(config)
        self.config = config

        self.vjepa2 = VJEPA2Model(config)

        self.decoder = VJEPA2Decoder(config, embeddings)

        self.post_init()

    def forward(
            self,
            pixel_values_videos: torch.Tensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_loss: bool = False
    ):
        encoder_outputs = self.vjepa2(
            pixel_values_videos=pixel_values_videos,
            skip_predictor=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        reconstructed_images = self.decoder(encoder_outputs.last_hidden_state)

        loss = None
        if return_loss:
            target_images = pixel_values_videos[:, 0]
            loss = nn.functional.mse_loss(reconstructed_images.unsqueeze(0), target_images)

        return {
            "reconstructed_images": reconstructed_images,
            "loss": loss,
            "hidden_states": encoder_outputs.hidden_states,
            "attentions": encoder_outputs.attentions
        }
