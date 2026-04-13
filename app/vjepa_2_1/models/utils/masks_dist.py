import torch


def _get_frame_pos(ids, H_patches=None, W_patches=None, grid_size=None):
    if H_patches is None or W_patches is None:
        tokens_per_frame = int(grid_size * grid_size)
    else:
        tokens_per_frame = int(H_patches * W_patches)
    return ids // tokens_per_frame


def _get_height_pos(ids, H_patches=None, W_patches=None, grid_size=None):
    # Remove frame component from ids
    if H_patches is None or W_patches is None:
        tokens_per_frame = int(grid_size * grid_size)
        tokens_per_row = grid_size
    else:
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
    frame_ids = _get_frame_pos(ids, H_patches, W_patches, grid_size)
    ids = ids - tokens_per_frame * frame_ids
    # --
    return ids // tokens_per_row


def separate_positions(ids, H_patches=None, W_patches=None, grid_size=None):
    if H_patches is None or W_patches is None:
        tokens_per_frame = int(grid_size * grid_size)
        tokens_per_row = grid_size
    else:
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
    frame_ids = _get_frame_pos(ids, H_patches, W_patches, grid_size)
    # --
    height_ids = _get_height_pos(ids, H_patches, W_patches, grid_size)
    # --
    # Remove frame component from ids (1st term) and height component (2nd term)
    width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
    return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids


def compute_mask_distance(masks_pred, masks_enc, grid_size, offset_context_loss):
    # masks_pred: [fpc][mask] where each mask is [B, N_pred]
    # masks_enc: [fpc][mask] where each mask is [B, N_enc]
    distances = []
    for masks_pred_i, masks_enc_i in zip(masks_pred, masks_enc):
        row_distances = []
        for masks_pred_ij, masks_enc_ij in zip(masks_pred_i, masks_enc_i):
            N_enc_tokens = masks_enc_ij.shape[1]
            d_enc, h_enc, w_enc = separate_positions(
                masks_enc_ij, grid_size=grid_size
            )  # (BS, N_enc)
            d_pred, h_pred, w_pred = separate_positions(
                masks_pred_ij, grid_size=grid_size
            )  # (BS, N_pred)
            pred = torch.stack([d_pred, h_pred, w_pred], dim=-1)  # (BS, N_pred, 3)
            enc_distances = []
            for enc_token in range(N_enc_tokens):
                enc_position = torch.stack(
                    [d_enc[:, enc_token], h_enc[:, enc_token], w_enc[:, enc_token]],
                    dim=-1,
                ).unsqueeze(
                    1
                )  # (BS, 1, 3)
                dist = torch.cdist(enc_position, pred, p=2)  # (BS, N_enc)
                dmin, argmin = dist.min(dim=-1)
                if offset_context_loss:
                    coeff = grid_size // 16  # Which is the default value of grid_size
                    dmin = dmin * (1.0 / coeff)
                dmin = dmin**0.5  # We want that it decreases less agressive
                enc_distances.append(dmin)
            enc_distances = torch.stack(enc_distances, dim=-1).squeeze()  # (BS, N_enc)
            row_distances.append(enc_distances)
        distances.append(row_distances)
    return distances
