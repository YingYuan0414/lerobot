#!/usr/bin/env python
"""DINOv2 RGB encoder for the diffusion policy.

Drop-in replacement for `DiffusionRgbEncoder` (ResNet18 + SpatialSoftmax) that
follows the DexDirect recipe (arXiv 2607.27784): a DINOv2 ViT-S/14 encoder over
a 240x240 downscale of the camera frame, random 224x224 crop + color jitter at
train time, center crop at eval.

The backbone is frozen by default and the 256 patch tokens (224/14 = 16 per
side) are reduced to a single vector by a trainable attention-pooling head. At
50-ish demos this is the safer choice: it keeps the ~21M backbone params out of
the gradient path so only the small pooling head has to be fit.

Interface contract with `modeling_diffusion.py`:
  - `.feature_dim`                     int, width of the returned vector
  - `.compute_crop_params(img_shape)`  -> {"roi_top", "roi_left", "top", "left"}
  - `.forward(x, crop_params=None)`    (B, C, H, W) in [0, 1] -> (B, feature_dim)

The full pipeline is: optional fixed ROI cut on the RAW frame -> resize to
`dinov2_resize_shape` -> `crop_shape` crop -> ImageNet normalize -> ViT. So the
two offset pairs live in different coordinate systems: `roi_top`/`roi_left` are
raw-camera pixels, `top`/`left` are post-resize pixels. Callers only ever
round-trip them back into `forward`, so the convention stays internal.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

# ImageNet statistics DINOv2 was trained with. Applied inside the encoder so the
# dataset-level VISUAL normalization can stay IDENTITY (i.e. plain [0, 1] input).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DiffusionDinoV2RgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector with a DINOv2 backbone."""

    def __init__(self, config):
        super().__init__()
        from transformers import AutoModel

        self.resize_shape = tuple(config.dinov2_resize_shape)
        self.crop_shape = tuple(config.crop_shape) if config.crop_shape is not None else None
        self.do_crop = self.crop_shape is not None
        self.crop_is_random = config.crop_is_random

        # Fixed region-of-interest, applied to the RAW frame before the resize.
        # A third-person camera typically spends most of its pixels on
        # background (people, monitors, the rest of the lab) that no action
        # depends on; cutting to the workspace first means the 240x240 resize
        # spends its resolution on the table instead. None = use the whole frame.
        self.roi = tuple(config.dinov2_roi) if config.dinov2_roi is not None else None
        if self.roi is not None and len(self.roi) != 4:
            raise ValueError(
                f"`dinov2_roi` must be (top, left, height, width). Got {self.roi}."
            )
        # Jitter the ROI itself at train time. This is the augmentation that
        # matters once the ROI is fixed: it stands in for small camera shifts,
        # which a fixed crop would otherwise make the policy brittle to.
        self.roi_jitter = int(config.dinov2_roi_jitter)

        self.backbone = AutoModel.from_pretrained(config.vision_backbone_path)
        self.patch_size = self.backbone.config.patch_size
        hidden_dim = self.backbone.config.hidden_size

        self.freeze_backbone = config.dinov2_freeze_backbone
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        # Input size actually fed to the ViT: the crop if cropping, else the resize.
        vit_h, vit_w = self.crop_shape if self.do_crop else self.resize_shape
        if vit_h % self.patch_size or vit_w % self.patch_size:
            raise ValueError(
                f"DINOv2 input {(vit_h, vit_w)} must be divisible by the patch size "
                f"{self.patch_size}. Adjust `crop_shape` / `dinov2_resize_shape`."
            )
        if self.do_crop and (
            self.crop_shape[0] > self.resize_shape[0] or self.crop_shape[1] > self.resize_shape[1]
        ):
            raise ValueError(
                f"`crop_shape` {self.crop_shape} must fit within `dinov2_resize_shape` "
                f"{self.resize_shape}."
            )

        # Attention pooling over patch tokens: one learnable query cross-attends
        # to the 256 patch tokens, so the head learns *where* to look instead of
        # averaging the card away into the background.
        self.num_heads = config.dinov2_pool_num_heads
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=self.num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self.feature_dim = config.dinov2_feature_dim
        self.out = nn.Linear(hidden_dim, self.feature_dim)
        self.relu = nn.ReLU()

        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode so its dropout/droppath stay off."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def compute_crop_params(self, img_shape: tuple) -> dict:
        """Sample the augmentation offsets for one camera.

        Two independent offsets are drawn, both zero-jitter at eval:
          * `roi_top`/`roi_left` shift the fixed ROI on the RAW frame by up to
            +/- `roi_jitter` px (kept inside the frame).
          * `top`/`left` are the 224-in-240 crop offsets, in RESIZED coordinates.

        `img_shape` is the raw (C, H, W) of the incoming frame, needed to keep a
        jittered ROI in bounds.
        """
        params = {"roi_top": 0, "roi_left": 0, "top": 0, "left": 0}
        jitter_on = self.training and self.crop_is_random

        if self.roi is not None and self.roi_jitter > 0:
            _, height, width = img_shape
            r_top, r_left, r_h, r_w = self.roi
            if jitter_on:
                j = self.roi_jitter
                # Clamp so the jittered ROI stays inside the frame.
                lo_t, hi_t = max(-j, -r_top), min(j, height - (r_top + r_h))
                lo_l, hi_l = max(-j, -r_left), min(j, width - (r_left + r_w))
                if hi_t >= lo_t:
                    params["roi_top"] = int(torch.randint(lo_t, hi_t + 1, (1,)).item())
                if hi_l >= lo_l:
                    params["roi_left"] = int(torch.randint(lo_l, hi_l + 1, (1,)).item())

        if self.do_crop:
            max_top = self.resize_shape[0] - self.crop_shape[0]
            max_left = self.resize_shape[1] - self.crop_shape[1]
            if not jitter_on:
                params["top"], params["left"] = max_top // 2, max_left // 2
            else:
                params["top"] = (
                    int(torch.randint(0, max_top + 1, (1,)).item()) if max_top > 0 else 0
                )
                params["left"] = (
                    int(torch.randint(0, max_left + 1, (1,)).item()) if max_left > 0 else 0
                )
        return params

    def forward(self, x: Tensor, crop_params: dict | None = None) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
            crop_params: offsets from `compute_crop_params`. If None, they are
                sampled here (random while training, centred otherwise).
        Returns:
            (B, feature_dim) image feature.
        """
        if crop_params is None:
            crop_params = self.compute_crop_params(tuple(x.shape[1:]))

        # Cut to the workspace on the RAW frame, before any downscaling, so the
        # resize below spends its 240x240 on the table rather than the room.
        if self.roi is not None:
            r_top, r_left, r_h, r_w = self.roi
            r_top += crop_params.get("roi_top", 0)
            r_left += crop_params.get("roi_left", 0)
            x = x[:, :, r_top : r_top + r_h, r_left : r_left + r_w]

        # Downscale the (possibly ROI-cut) frame to the working size.
        if x.shape[-2:] != self.resize_shape:
            x = F.interpolate(x, size=self.resize_shape, mode="bilinear", align_corners=False)

        if self.do_crop:
            top, left = crop_params["top"], crop_params["left"]
            crop_h, crop_w = self.crop_shape
            x = x[:, :, top : top + crop_h, left : left + crop_w]

        x = (x - self._mean) / self._std

        # Patch tokens only: index 0 of DINOv2's sequence is the CLS token, and
        # the spatial tokens are what the pooling head needs to localize on.
        if self.freeze_backbone:
            with torch.no_grad():
                tokens = self.backbone(pixel_values=x).last_hidden_state[:, 1:]
        else:
            tokens = self.backbone(pixel_values=x).last_hidden_state[:, 1:]

        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn_pool(query, tokens, tokens, need_weights=False)
        pooled = self.norm(pooled.squeeze(1))
        return self.relu(self.out(pooled))
