# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type, List

from .common import LayerNorm2d
from .multimodal_prompt import MultimodalPromptEncoder


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
        use_multimodal_prompt: bool = False,
        clip_model_path: Optional[str] = None,
        num_classes: int = 8,
        text_embed_dim: Optional[int] = None,  # 新增：CLIP/CoOp 输出的文本维度
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
          use_multimodal_prompt (bool): Whether to use multimodal prompts (CLIP-based)
          clip_model_path (str): Path to CLIP pretrained model
          num_classes (int): Number of classes for global classification
          text_embed_dim (int, optional): CLIP/CoOp 输出的文本维度 (例如 RN50 是 1024, ViT-B/16 是 512)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.use_multimodal_prompt = use_multimodal_prompt
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)
        
        # --- 新增：文本特征投影层 ---
        # 将 CLIP/CoOp 的文本特征维度映射到 SAM 的 embed_dim (256)
        # 如果提供了 text_embed_dim，则创建投影层；否则设为 None，在 forward 中动态创建
        if text_embed_dim is not None:
            self.text_projection = nn.Linear(text_embed_dim, embed_dim)
        else:
            self.text_projection = None
        
        # 多模态提示编码器（使用SAM ViT）
        if use_multimodal_prompt:
            # image_encoder将在SAM模型初始化后设置
            self.multimodal_encoder = MultimodalPromptEncoder(
                embed_dim=embed_dim,
                clip_model_path=clip_model_path,
                use_global_features=True,
                num_classes=num_classes,
                image_encoder=None,  # 将在SAM初始化后设置
            )
        else:
            self.multimodal_encoder = None

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel

        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)  #B,N+1,2
            labels = torch.cat([labels, padding_label], dim=1)


        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)  #B,N+1,256
        point_embedding[labels == -1] = 0.0

        self.not_a_point_embed.weight = torch.nn.Parameter(self.not_a_point_embed.weight.to(point_embedding.dtype), requires_grad=True)  # todo
        self.point_embeddings[0].weight = torch.nn.Parameter(self.point_embeddings[0].weight.to(point_embedding.dtype), requires_grad=True) #todo
        self.point_embeddings[1].weight = torch.nn.Parameter(self.point_embeddings[1].weight.to(point_embedding.dtype), requires_grad=True) #todo

        point_embedding[labels == -1] += self.not_a_point_embed.weight 
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""

        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        text_prompts: Optional[List[str]] = None,
        image_features: Optional[torch.Tensor] = None,
        global_labels: Optional[torch.Tensor] = None,
        raw_image: Optional[torch.Tensor] = None,
        image_encoder: Optional[torch.nn.Module] = None,
        text_embeddings: Optional[torch.Tensor] = None,  # 新增：来自 CoOp/PNuRL 的文本嵌入
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed
          text_prompts (List[str] or none): text prompts for multimodal encoding
          image_features (torch.Tensor or none): image features for multimodal encoding
          global_labels (torch.Tensor or none): global labels for multimodal encoding
          text_embeddings (torch.Tensor or none): 来自 CoOp/PNuRL 的文本嵌入 [B, L, C] 或 [B, C]
            L 是文本 token 长度，C 是 CLIP 维度

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        # 如果只有文本提示，需要根据文本的 batch size 确定
        if bs == 1 and text_embeddings is not None:
            bs = text_embeddings.shape[0]
        
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device()) #B,0,256  空[]

        if points is not None:
            coords, labels = points     #coords:B,N,2  labels:B,N
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        # --- 新增：处理文本嵌入（来自 CoOp/PNuRL）---
        if text_embeddings is not None:
            # text_embeddings shape 假设为 [B, L, C] 或 [B, C]
            # L 是文本 token 长度，C 是 CLIP 维度
            if text_embeddings.dim() == 2:
                text_embeddings = text_embeddings.unsqueeze(1)  # [B, 1, C]
            
            # 获取文本维度
            text_dim = text_embeddings.shape[-1]
            
            # 如果投影层不存在或维度不匹配，动态创建
            if self.text_projection is None or self.text_projection.in_features != text_dim:
                device = text_embeddings.device
                self.text_projection = nn.Linear(text_dim, self.embed_dim).to(device)
            
            # 投影到 SAM 维度
            text_embeddings_proj = self.text_projection(text_embeddings)  # [B, L, embed_dim]
            
            # 拼接到 sparse_embeddings (SAM 将文本视为类似点/框的 token)
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings_proj], dim=1)

        # 多模态提示融合（使用SAM ViT特征）
        if self.use_multimodal_prompt and self.multimodal_encoder is not None:
            if image_features is not None:
                multimodal_embed = self.multimodal_encoder(
                    image_features=image_features,
                    text_prompts=text_prompts,
                    global_labels=global_labels,
                    raw_image=raw_image,
                    image_encoder=image_encoder,
                )  # [B, embed_dim]
                # 将多模态嵌入添加到sparse_embeddings
                multimodal_embed = multimodal_embed.unsqueeze(1)  # [B, 1, embed_dim]
                sparse_embeddings = torch.cat([sparse_embeddings, multimodal_embed], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        # coords = coords @ self.positional_encoding_gaussian_matrix
        coords = coords @ self.positional_encoding_gaussian_matrix.to(torch.float32) # todo
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size

        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]

        return self._pe_encoding(coords.to(torch.float))  # B x N x C
