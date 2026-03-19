import torch
import torch.nn.functional as F


def _get_sobel_kernels(device, dtype):
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)


def msge_loss(true, pred, focus):
    """
    Mean Squared Error of Gradients (MSGE) loss for HoVer-style HV maps.

    Args:
        true:  [B, 2, H, W] ground-truth HV map (V,H).
        pred:  [B, 2, H, W] predicted HV map (V,H).
        focus: [B, H, W] or [B, 1, H, W] binary mask; loss computed only in focus==1.
    """
    if focus.dim() == 3:
        focus = focus.unsqueeze(1)
    focus = focus.float()

    # ensure float32 for stable gradients
    true = true.float()
    pred = pred.float()

    kx, ky = _get_sobel_kernels(device=pred.device, dtype=pred.dtype)

    # depthwise conv for each channel (2 channels)
    kx2 = kx.repeat(2, 1, 1, 1)
    ky2 = ky.repeat(2, 1, 1, 1)

    true_gx = F.conv2d(true, kx2, padding=1, groups=2)
    true_gy = F.conv2d(true, ky2, padding=1, groups=2)
    pred_gx = F.conv2d(pred, kx2, padding=1, groups=2)
    pred_gy = F.conv2d(pred, ky2, padding=1, groups=2)

    diff = (true_gx - pred_gx) ** 2 + (true_gy - pred_gy) ** 2
    diff = diff * focus  # broadcast over channel
    denom = focus.sum() * diff.shape[1] + 1e-8
    return diff.sum() / denom


@torch.no_grad()
def generate_hv_map_from_inst(inst_map: torch.Tensor) -> torch.Tensor:
    """
    Build HoVer-style HV map from an instance ID map.

    Args:
        inst_map: [H, W] int tensor, 0=background, >0=instance id.
    Returns:
        hv: [2, H, W] float tensor in [-1, 1] inside instances, 0 outside.
             hv[0]=V (y-direction), hv[1]=H (x-direction)
    """
    if inst_map.dim() != 2:
        raise ValueError(f"inst_map must be [H,W], got {tuple(inst_map.shape)}")

    device = inst_map.device
    inst = inst_map.to(torch.int64).cpu().numpy()
    H, W = inst.shape

    hv_v = torch.zeros((H, W), dtype=torch.float32)
    hv_h = torch.zeros((H, W), dtype=torch.float32)

    ids = sorted([int(x) for x in set(inst.reshape(-1).tolist()) if x > 0])
    for ins_id in ids:
        ys, xs = (inst == ins_id).nonzero()
        if ys.size == 0:
            continue

        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

        # centroid (in float)
        cy = float(ys.mean())
        cx = float(xs.mean())

        # normalization radius (avoid div0 for thin objects)
        ry = max(1.0, (y1 - y0 + 1) / 2.0)
        rx = max(1.0, (x1 - x0 + 1) / 2.0)

        ys_t = torch.from_numpy(ys).float()
        xs_t = torch.from_numpy(xs).float()

        # V: vertical distance (y), H: horizontal distance (x)
        v = (ys_t - cy) / ry
        h = (xs_t - cx) / rx

        hv_v[ys_t.long(), xs_t.long()] = v.clamp(-1.0, 1.0)
        hv_h[ys_t.long(), xs_t.long()] = h.clamp(-1.0, 1.0)

    hv = torch.stack([hv_v, hv_h], dim=0).to(device=device)
    return hv

