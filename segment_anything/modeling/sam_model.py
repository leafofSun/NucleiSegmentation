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
        if use_pnurl:
            if pnurl_config is None:
                pnurl_config = {}
            # SAM ViT 的 out_chans 在初始化时设置，通常是 256
            # 从 image_encoder 的 neck 层获取输出通道数
            # neck 的第一个 Conv2d 的输出通道数就是 out_chans
            if hasattr(image_encoder, 'neck') and len(image_encoder.neck) > 0:
                first_conv = image_encoder.neck[0]
                if isinstance(first_conv, nn.Conv2d):
                    feat_dim = first_conv.out_channels
                else:
                    feat_dim = 256
            else:
                # 默认使用 256（SAM 的标准输出维度）
                feat_dim = 256
            self.pnurl = PNuRL(
                feat_dim=feat_dim,
                embed_dim=256,
                clip_model_path=pnurl_config.get('clip_model_path', None),
                num_classes_per_attr=pnurl_config.get('num_classes_per_attr', [3, 5, 4, 3, 3]),
                attr_loss_weight=pnurl_config.get('attr_loss_weight', 1.0),
            )
            
            # 初始化 prompt_encoder 的 text_projection（用于 PNuRL 的 learnable_context）
            # PNuRL 的 learnable_context 输出维度是 feat_dim (通常是 256)
            if self.prompt_encoder.text_projection is None:
                self.prompt_encoder.text_projection = nn.Linear(feat_dim, self.prompt_encoder.embed_dim)
                print(f"✓ 已初始化 PromptEncoder.text_projection 用于 PNuRL (输入维度: {feat_dim})")
        else:
            self.pnurl = None
        
        # 初始化 CoOp/PNuRL 文本编码器（用于生成可学习的文本提示）
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
            
            # 获取文本嵌入维度，并更新 prompt_encoder 的 text_projection
            text_embed_dim = self.pnurl_text_encoder.get_text_embed_dim()
            # 如果 prompt_encoder 还没有 text_projection，创建一个
            if self.prompt_encoder.text_projection is None:
                self.prompt_encoder.text_projection = nn.Linear(text_embed_dim, self.prompt_encoder.embed_dim)
        else:
            self.pnurl_text_encoder = None

    @property
    def device(self) -> Any:
        return self.pixel_mean.device
 
    def forward(self, batched_input: Dict[str, Any], multimask_output: bool) -> List[Dict[str, torch.Tensor]]:

        input_images = batched_input.get("image")
        image_embeddings = self.image_encoder(input_images)
        
        # PNuRL 处理（如果启用）：使用属性提示词对ViT特征进行加权
        pnurl_loss = None
        pnurl_context = None
        if self.use_pnurl and self.pnurl is not None:
            attribute_prompts = batched_input.get("attribute_prompts", None)
            attribute_labels = batched_input.get("attribute_labels", None)
            return_loss = self.training and attribute_labels is not None
            
            # PNuRL返回：加权后的ViT特征、可学习上下文、损失、logits
            weighted_image_embeddings, pnurl_context, pnurl_loss, _ = self.pnurl(
                image_features=image_embeddings,
                attribute_prompts=attribute_prompts,
                attribute_labels=attribute_labels,
                return_loss=return_loss,
            )
            # 使用加权后的ViT特征替代原始特征（关键：让PNuRL的加权生效）
            image_embeddings = weighted_image_embeddings

        if "point_coords" in batched_input and batched_input["point_coords"] != None:
            points = (batched_input["point_coords"], batched_input["point_labels"])
        else:
            points = None

        # 支持多模态提示（使用SAM ViT特征）
        text_prompts = batched_input.get("text_prompts", None)
        global_labels = batched_input.get("global_labels", None)
        # 如果使用多模态提示，使用图像编码器的输出（已经是ViT特征）
        image_features_for_prompt = batched_input.get("image_features", None)
        if image_features_for_prompt is None and hasattr(self.prompt_encoder, 'use_multimodal_prompt') and self.prompt_encoder.use_multimodal_prompt:
            # 使用SAM ViT编码器的输出（已经是强大的ViT特征）
            image_features_for_prompt = image_embeddings
        
        # 设置多模态编码器的image_encoder引用（如果还没有设置）
        if hasattr(self.prompt_encoder, 'multimodal_encoder') and self.prompt_encoder.multimodal_encoder is not None:
            if self.prompt_encoder.multimodal_encoder.clip_vit.image_encoder is None:
                self.prompt_encoder.multimodal_encoder.clip_vit.set_image_encoder(self.image_encoder)
        
        # CoOp/PNuRL 文本提示（可学习的语义提示）
        text_embeddings = None
        if self.use_coop_prompt and self.pnurl_text_encoder is not None:
            # 获取目标类别索引（从 batched_input 中获取，例如 "target_class_idx"）
            target_class_idx = batched_input.get("target_class_idx", None)
            if target_class_idx is not None:
                # 将 target_class_idx 转换为 tensor（如果还不是）
                if not isinstance(target_class_idx, torch.Tensor):
                    target_class_idx = torch.tensor(target_class_idx, device=self.device)
                # 生成可学习的文本嵌入
                text_embeddings = self.pnurl_text_encoder(target_class_idx)  # [B, text_embed_dim]
                # 添加一个维度以匹配 sparse_embeddings 的格式 [B, 1, text_embed_dim]
                text_embeddings = text_embeddings.unsqueeze(1)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
            text_prompts=text_prompts,
            image_features=image_features_for_prompt,
            global_labels=global_labels,
            raw_image=input_images,  # 传递原始图像，以便CLIPViT可以使用
            image_encoder=self.image_encoder,  # 传递SAM的image_encoder
            text_embeddings=text_embeddings,  # 新增：来自 CoOp/PNuRL 的文本嵌入
        )  # sparse_embeddings:[2, 3, 256],  dense_embeddings:[2, 256, 64, 64]

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),  # 1x(256)x(64)x(64)
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

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
        
        # 添加 PNuRL 相关输出
        if self.use_pnurl:
            outputs["pnurl_loss"] = pnurl_loss
            outputs["pnurl_context"] = pnurl_context

        return outputs

    def postprocess_masks(self,masks: torch.Tensor, input_size: Tuple[int, ...],original_size: Tuple[int, ...],) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size), mode="bilinear", align_corners=False,)  #[1,1024,1024]

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
