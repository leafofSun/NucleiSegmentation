# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple, Optional

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .pnurl import PNuRL


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        use_pnurl: bool = False,
        pnurl_config: Optional[Dict[str, Any]] = None,
        use_coop_prompt: bool = False,
        coop_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.use_pnurl = use_pnurl
        self.use_coop_prompt = use_coop_prompt
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        # 初始化 PNuRL 模块
        if self.use_pnurl and pnurl_config is not None:
            # PNuRL 期望的 feat_dim 通常为 256 (与 prompt_encoder.embed_dim 一致)
            self.pnurl = PNuRL(
                feat_dim=prompt_encoder.embed_dim,
                embed_dim=prompt_encoder.embed_dim,
                clip_model_path=pnurl_config.get('clip_model_path'),
                num_classes_per_attr=pnurl_config.get('num_classes_per_attr', [3, 5, 4, 3, 3]),
                attr_loss_weight=pnurl_config.get('attr_loss_weight', 1.0)
            )
            # 初始化 prompt_encoder 的 text_projection（用于 PNuRL 的 learnable_context）
            if self.prompt_encoder.text_projection is None:
                self.prompt_encoder.text_projection = nn.Linear(prompt_encoder.embed_dim, prompt_encoder.embed_dim)
        else:
            self.pnurl = None

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """

        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        if masks.dim() == 3:
          masks = masks.unsqueeze(0)
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

# =============================================================================
# [新增] 引入依赖库
# =============================================================================
import torch.nn as nn
import sys
import os

# 1. 尝试导入 CLIP
try:
    import clip
except ImportError:
    # 如果环境里没装，尝试去上一级目录找 (假设你把 CLIP 放到了项目旁边的文件夹)
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../CLIP")) 
    try:
        import clip
    except ImportError:
        print("⚠️ Warning: CLIP not found. Text encoder will fail.")

# 2. 尝试导入你的 Prompt Generator
# 假设 prompt_generator.py 放在项目根目录
try:
    from prompt_generator import TextGuidedPointGenerator
except ImportError:
    # 这一步是为了让 python 能找到根目录下的 prompt_generator
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
    try:
        from prompt_generator import TextGuidedPointGenerator
    except ImportError:
        print("⚠️ Warning: prompt_generator.py not found. Please check file path.")

# =============================================================================
# [新增] TextSam 类 (End-to-End Multi-modal SAM)
# =============================================================================
class TextSam(Sam):
    def __init__(
        self, 
        image_encoder, 
        prompt_encoder, 
        mask_decoder, 
        clip_model_name="ViT-B/16", 
        text_dim=512, 
        embed_dim=256
    ):
        super().__init__(image_encoder, prompt_encoder, mask_decoder)
        
        # 冻结 SAM 参数
        for param in self.parameters():
            param.requires_grad = False
            
        # 解冻 Mask Decoder
        for param in self.mask_decoder.parameters():
            param.requires_grad = True
            
        self.prompt_generator = TextGuidedPointGenerator(embed_dim=embed_dim, text_dim=text_dim)
        
        # 加载 CLIP (Dummy embedding for simplicity if not using real text)
        # 这里假设 prompt_generator 内部已经处理好了，或者我们传入固定的 embedding
        # 为了简化，我们在 forward 里生成 dummy text embedding
        self.register_buffer("text_embedding", torch.randn(1, 2, text_dim)) # [1, N_Class, Dim]

    def forward(self, batched_input, multimask_output=True):
        # 1. Image Encoder
        input_images = torch.stack([x["image"] for x in batched_input], dim=0)
        # input_images is [B, 3, 256, 256] (0-255)
        
        input_images_processed = self.preprocess(input_images)
        image_embeddings = self.image_encoder(input_images_processed) # [B, 256, 64, 64]
        
        # 2. Prompt Generator
        # 使用 dummy text embedding (2 classes: nuclei, bg)
        # 实际项目中应该传入真实的 CLIP embedding
        B = len(batched_input)
        text_embed = self.text_embedding.expand(B, -1, -1) # [B, 2, 512]
        
        heatmap_logits = self.prompt_generator(image_embeddings, text_embed)
        
        # 3. Get Points from Heatmap
        points, labels = self.prompt_generator.get_points_from_heatmap(heatmap_logits, topk=1)
        # points shape: [B, K, 2], coordinates in Feature Map scale (e.g., 0-64)
        
        # === 🚨 CRITICAL FIX: Rescale Points to Input Image Size ===
        # 特征图大小
        feat_h, feat_w = image_embeddings.shape[-2:] # 64, 64
        # 输入图片大小 (padding 前)
        input_h, input_w = input_images.shape[-2:]   # 256, 256
        
        # 计算缩放比例
        scale_x = input_w / feat_w
        scale_y = input_h / feat_h
        
        # 映射坐标到原图尺度 (并加 0.5 居中)
        points_rescaled = points.clone()
        points_rescaled[:, :, 0] = points[:, :, 0] * scale_x + (scale_x / 2)
        points_rescaled[:, :, 1] = points[:, :, 1] * scale_y + (scale_y / 2)
        # ==========================================================
        
        outputs = []
        for i, curr_embedding in enumerate(image_embeddings):
            # 构造 SAM 需要的点提示格式 (B, N, 2)
            # points_rescaled[i] is [K, 2]
            point_coords = points_rescaled[i].unsqueeze(0) # [1, K, 2]
            point_labels = labels[i].unsqueeze(0)          # [1, K]
            
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            
            # Postprocess
            # input_size should be the size before padding, i.e., 256x256
            input_size = batched_input[i]["image"].shape[-2:]
            original_size = batched_input[i]["original_size"]
            
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=input_size,
                original_size=original_size,
            )
            
            outputs.append({
                "masks": masks,
                "iou_predictions": iou_predictions,
                "heatmap_logits": heatmap_logits[i].unsqueeze(0)
            })
            
        return outputs