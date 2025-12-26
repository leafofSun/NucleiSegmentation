# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple, Optional
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .pnurl import PNuRL
import sys
import os

try:
    import clip
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../CLIP")) 
    try:
        import clip
    except ImportError:
        print("⚠️ Warning: CLIP not found.")

try:
    from prompt_generator import TextGuidedPointGenerator
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
    try:
        from prompt_generator import TextGuidedPointGenerator
    except ImportError:
        print("⚠️ Warning: prompt_generator.py not found.")

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
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.use_pnurl = use_pnurl
        self.use_coop_prompt = use_coop_prompt
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        if self.use_pnurl and pnurl_config is not None:
            self.pnurl = PNuRL(
                feat_dim=prompt_encoder.embed_dim,
                embed_dim=prompt_encoder.embed_dim,
                clip_model_path=pnurl_config.get('clip_model_path'),
                num_classes_per_attr=pnurl_config.get('num_classes_per_attr', [3, 5, 4, 3, 3]),
                attr_loss_weight=pnurl_config.get('attr_loss_weight', 1.0)
            )
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
        # [Fix] 这里的归一化是必要的，但必须确保输入是 0-255 的 Tensor
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


class TextSam(Sam):
    def __init__(
        self, 
        image_encoder, 
        prompt_encoder, 
        mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        clip_model_name="ViT-B/16",
        text_dim=512,
        embed_dim=256
    ):
        super().__init__(image_encoder, prompt_encoder, mask_decoder, pixel_mean, pixel_std)
        
        print(f"Loading CLIP model: {clip_model_name}...")
        clip_model, _ = clip.load(clip_model_name, device="cpu")
        bi_prompts = ["Nuclei", "Background"]
        text_tokens = clip.tokenize(bi_prompts)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.float()
        self.register_buffer("text_features_static", text_features.unsqueeze(0)) 
        del clip_model 

        self.prompt_generator = TextGuidedPointGenerator(
            embed_dim=embed_dim,
            text_dim=text_dim
        )
        
        # 冻结策略
        for param in self.image_encoder.parameters(): param.requires_grad = False
        for param in self.prompt_encoder.parameters(): param.requires_grad = False
        for param in self.mask_decoder.parameters(): param.requires_grad = True
        
        # 解冻 Adapter
        adapter_count = 0
        for name, param in self.image_encoder.named_parameters():
            if "Adapter" in name:
                param.requires_grad = True
                adapter_count += 1
        print(f"✅ TextSam Initialized: {adapter_count} Adapter Layers Unfrozen.")

    def forward(self, batched_input, multimask_output=False):
        # 1. 图像编码
        # 注意：这里的 input_images 是 padding 后的（例如 1024x1024）
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images) 

        # 2. 文本特征
        B = len(batched_input)
        text_features = self.text_features_static.expand(B, -1, -1) 

        # 3. 热力图
        heatmap_logits = self.prompt_generator(image_embeddings, text_features)
        
        # 4. 提取点 (Feature Map Scale)
        points_in_feat, point_labels = self.prompt_generator.get_points_from_heatmap(heatmap_logits, topk=1)
        
        # 5. 坐标映射 (Feature Map -> Image Scale)
        # [CRITICAL FIX]
        # 特征图大小: 64
        feat_size = image_embeddings.shape[-1] 
        # 输入图大小: 1024 (ImageEncoderViT 的默认输入尺寸)
        input_size = self.image_encoder.img_size 
        
        # 计算缩放倍率 (1024 / 64 = 16)
        scale_factor = input_size / feat_size
        
        # 映射坐标并居中
        point_coords = (points_in_feat * scale_factor) + (scale_factor * 0.5)

        # 6. SAM 解码
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
        
        # 7. 结果封装
        outputs = []
        for i in range(len(batched_input)):
            # 后处理: 还原到原始图片的实际大小 (例如 256x256)
            mask_post = self.postprocess_masks(
                low_res_masks[i],
                input_size=batched_input[i]["image"].shape[-2:], 
                original_size=batched_input[i]["original_size"],
            )
            
            outputs.append({
                "masks": mask_post,
                "iou_predictions": iou_predictions[i],
                "low_res_masks": low_res_masks[i],
                "heatmap_logits": heatmap_logits[i].unsqueeze(0)
            })
            
        return outputs