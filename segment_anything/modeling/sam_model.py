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
from .pnurl_text_encoder import PNuRLTextEncoder


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

        # 1. 初始化 PNuRL 模块
        if use_pnurl:
            if pnurl_config is None:
                pnurl_config = {}
            if hasattr(image_encoder, 'neck') and len(image_encoder.neck) > 0:
                first_conv = image_encoder.neck[0]
                feat_dim = first_conv.out_channels if isinstance(first_conv, nn.Conv2d) else 256
            else:
                feat_dim = 256
                
            self.pnurl = PNuRL(
                embed_dim=feat_dim,  
                text_dim=256,        
                num_classes_per_attr=pnurl_config.get('num_classes_per_attr', [2, 3, 2, 3, 3]),
                attr_loss_weight=pnurl_config.get('attr_loss_weight', 1.0),
            )
            
            if self.prompt_encoder.text_projection is None:
                self.prompt_encoder.text_projection = nn.Linear(feat_dim, self.prompt_encoder.embed_dim)
        else:
            self.pnurl = None

        # 2. 初始化 CoOp/PNuRL 文本编码器
        if use_coop_prompt:
            if coop_config is None:
                coop_config = {}
            
            classnames = coop_config.get('classnames', ['Nuclei', 'Cell', 'Tissue'])
            clip_model_name = coop_config.get('clip_model_name', 'ViT-B/16')
            clip_model_path = coop_config.get('clip_model_path', None)
            n_ctx = coop_config.get('n_ctx', 16)
            ctx_init = coop_config.get('ctx_init', None)
            
            self.pnurl_text_encoder = PNuRLTextEncoder(
                classnames=classnames,
                clip_model_name=clip_model_name,
                clip_model_path=clip_model_path,
                n_ctx=n_ctx,
                ctx_init=ctx_init,
            )
            
            text_embed_dim = self.pnurl_text_encoder.get_text_embed_dim()
            if self.prompt_encoder.text_projection is None:
                self.prompt_encoder.text_projection = nn.Linear(text_embed_dim, self.prompt_encoder.embed_dim)
        else:
            self.pnurl_text_encoder = None

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, batched_input: Dict[str, Any], multimask_output: bool) -> List[Dict[str, torch.Tensor]]:
        # --- 1. 图像特征提取 ---
        input_images = batched_input.get("image")
        image_embeddings = self.image_encoder(input_images)
        
        cnn_feat_s1 = batched_input.get("cnn_feat_s1", None)
        cnn_feat_s2 = batched_input.get("cnn_feat_s2", None)

        # --- 2. 文本特征编码 (CoOp) ---
        text_embeddings = None
        if self.use_coop_prompt and self.pnurl_text_encoder is not None:
            target_class_idx = batched_input.get("target_class_idx", None)
            if target_class_idx is not None:
                if not isinstance(target_class_idx, torch.Tensor):
                    target_class_idx = torch.tensor(target_class_idx, device=self.device)
                text_embeddings = self.pnurl_text_encoder(target_class_idx)  # [B, text_embed_dim]
                # 添加一个维度以匹配 sparse_embeddings 的格式 [B, 1, text_embed_dim]
                text_embeddings = text_embeddings.unsqueeze(1)

        # --- 3. 频域解耦路由 (PNuRL) ---
        pnurl_loss = None
        pnurl_context = None
        density_map = None
        txt_attr_feat = None
        txt_mor_feat = None
        logits_dict = None

        if self.use_pnurl and self.pnurl is not None:
            attribute_labels = batched_input.get("attr_labels", batched_input.get("attribute_labels", None))
            return_loss = self.training and attribute_labels is not None
            
            # 🔥 核心修正：正确接收 PNuRL 的 7 个解耦输出
            refined_features, pnurl_context, pnurl_loss, logits_dict, density_map, txt_attr_feat, txt_mor_feat = self.pnurl(
                image_features=image_embeddings,
                text_embed=text_embeddings.squeeze(1) if text_embeddings is not None else None, 
                attribute_labels=attribute_labels,
                return_loss=return_loss,
            )
            # 使用加权后的特征替代原始特征
            image_embeddings = refined_features
        else:
            # 如果不启用 PNuRL，Fallback 到纯文本嵌入
            pnurl_context = text_embeddings.squeeze(1) if text_embeddings is not None else None

        # --- 4. 稀疏/密集提示编码 ---
        if "point_coords" in batched_input and batched_input["point_coords"] is not None:
            points = (batched_input["point_coords"], batched_input["point_labels"])
        else:
            points = None

        # 调整上下文维度以适配 PromptEncoder
        if pnurl_context is not None and pnurl_context.dim() == 2:
            pnurl_context = pnurl_context.unsqueeze(1) # [B, 1, C]

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
            text_embeddings=pnurl_context, 
        ) 

        # --- 5. 双流引导解码 (ASR + MaskDecoder) ---
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),  
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            cnn_feat_s1=cnn_feat_s1,
            cnn_feat_s2=cnn_feat_s2,
            attr_prompt=txt_attr_feat,  # 🟢 低频路由 -> 宏观属性
            morph_feat=txt_mor_feat,    # 🔴 高频路由 -> 微观形态
        )

        # --- 6. 后处理与输出 ---
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=batched_input["image"].shape[-2:],
            original_size=batched_input["original_size"],
        )

        outputs = {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
        }
        
        # 返回辅助训练信息 (给 train.py 算 loss 用)
        if self.use_pnurl:
            outputs["pnurl_loss"] = pnurl_loss
            outputs["pnurl_context"] = pnurl_context
            outputs["density_map"] = density_map
            outputs["attr_logits"] = logits_dict

        return outputs

    def postprocess_masks(self, masks: torch.Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...]) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size), mode="bilinear", align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x