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
    """
    支持双向语义引导（Bi-directional Semantic Prompting）的 SAM 变体。
    它会自动生成 "Nuclei" (正向) 和 "Background" (负向) 的提示点，混合输入给 SAM。
    """
    def __init__(
        self, 
        image_encoder, 
        prompt_encoder, 
        mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        clip_model_name="ViT-B/16", # 新增参数
        text_dim=512,               # 新增参数
        embed_dim=256               # 新增参数
    ):
        super().__init__(image_encoder, prompt_encoder, mask_decoder, pixel_mean, pixel_std)
        
        # 1. 初始化 CLIP (冻结参数，只用于提取文本特征)
        print(f"Loading CLIP model: {clip_model_name}...")
        self.clip_model, self.preprocess_clip = clip.load(clip_model_name, device="cpu")
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 2. 初始化 提示生成器 (我们刚刚修改过的双通道生成器)
        self.prompt_generator = TextGuidedPointGenerator(
            embed_dim=embed_dim,
            text_dim=text_dim
        )

    def forward(self, batched_input, multimask_output=False):
        # --- A. 图像编码 ---
        # 预处理图片并计算 Image Embeddings
        processed_images = [self.preprocess_input(x["image"]) for x in batched_input]
        input_images = torch.stack(processed_images).to(self.device)
        image_embeddings = self.image_encoder(input_images) # [B, 256, 64, 64]

        # --- B. 构造双向文本 (Bi-directional Text Construction) ---
        # 强制构造 [Nuclei, Background] 语义对
        # 这里的 "Nuclei" 对应 Channel 0 (正样本), "Background" 对应 Channel 1 (负样本)
        B = len(batched_input)
        bi_prompts = ["Nuclei", "Background"] 
        
        # 使用 CLIP 编码文本
        text_tokens = clip.tokenize(bi_prompts).to(self.device) # [2, 77]
        with torch.no_grad():
            text_features_proto = self.clip_model.encode_text(text_tokens) # [2, 512]
            text_features_proto = text_features_proto / text_features_proto.norm(dim=-1, keepdim=True)
            text_features_proto = text_features_proto.float()
            # 扩展到 Batch 维度: [B, 2, 512]
            text_features = text_features_proto.unsqueeze(0).repeat(B, 1, 1)

        # --- C. 提示生成 (Generator Forward) ---
        # 生成双通道热力图: [B, 2, 64, 64]
        heatmap_logits = self.prompt_generator(image_embeddings, text_features)
        
        # 提取点: 每个通道提取最强点 (Top-1)
        # points_in_feat: [B, 2, 2] (坐标)
        # point_labels:   [B, 2]    (1 for Nuclei, 0 for Background)
        points_in_feat, point_labels = self.prompt_generator.get_points_from_heatmap(heatmap_logits, topk=1)
        
        # --- D. 坐标映射 (Feature Map -> Image Size) ---
        # 从 64x64 映射回 1024x1024 (Scale factor usually 16)
        scale_factor = input_images.shape[-1] / image_embeddings.shape[-1]
        point_coords = points_in_feat * scale_factor

        # --- E. SAM 解码 ---
        # 将提取到的正负点混合喂给 Prompt Encoder
        # SAM 会自动处理 Label=0 的点作为"排除区域"
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        
        # --- F. 结果封装 ---
        outputs = []
        for i in range(len(batched_input)):
            # 后处理 Mask 到原始尺寸
            mask_post = self.postprocess_masks(
                low_res_masks[i],
                input_size=input_images.shape[-2:],
                original_size=batched_input[i]["original_size"],
            )
            
            out_dict = {
                "masks": mask_post,
                "iou_predictions": iou_predictions[i],
                "low_res_masks": low_res_masks[i],
                "heatmap_logits": heatmap_logits[i] # 关键：返回用于 Loss 计算
            }
            outputs.append(out_dict)
            
        return outputs

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        # 辅助函数：对输入图像进行归一化和 Padding
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = nn.functional.pad(x, (0, padw, 0, padh))
        return x