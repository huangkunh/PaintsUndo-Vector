"""
Transformer 模型 - 基于 GPT 结构的自回归 Action 预测

架构:
1. 图像编码器 (CNN → 特征向量)
2. Token Embedding + Positional Encoding
3. GPT Decoder (自回归预测下一个 Token)
4. 输出投影 (预测 Token ID)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from engine_reverser.tokenizer import ActionTokenizer, PAD, END_SEQUENCE, vocab_size


class ImageEncoder(nn.Module):
    """图像特征提取器 (轻量 CNN)"""
    
    def __init__(self, output_dim: int = 512):
        super().__init__()
        # 简单 CNN: 640x480 → feature vector
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=4, padding=3),  # 160x120
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=4, padding=2),  # 40x30
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 20x15
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 10x8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] 输入图像
        Returns:
            [B, output_dim] 图像特征
        """
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        return x + self.pe[:, :x.size(1)]


class StrokeTransformer(nn.Module):
    """
    笔画重建 Transformer
    
    输入: 图像特征 + 已生成的 Token 序列
    输出: 下一个 Token 的概率分布
    """
    
    def __init__(
        self,
        vocab_size: int = 500000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        image_feature_dim: int = 512,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 图像特征投影
        self.image_proj = nn.Linear(image_feature_dim, d_model)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 图像编码器
        self.image_encoder = ImageEncoder(output_dim=image_feature_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """生成因果注意力掩码"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask
    
    def forward(
        self,
        image: torch.Tensor,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W] 输入图像
            tokens: [B, T] 已生成的 Token 序列
            padding_mask: [B, T] padding 掩码
        Returns:
            [B, T, vocab_size] 每个位置的 Token 预测
        """
        B, T = tokens.shape
        device = tokens.device
        
        # 图像特征
        img_feat = self.image_encoder(image)  # [B, image_feature_dim]
        img_feat = self.image_proj(img_feat)   # [B, d_model]
        
        # Token Embedding + 位置编码
        token_emb = self.token_embedding(tokens)  # [B, T, d_model]
        token_emb = self.pos_encoding(token_emb)
        token_emb = self.dropout(token_emb)
        
        # 图像特征作为 memory (扩展为序列)
        memory = img_feat.unsqueeze(1).expand(B, T, self.d_model)  # [B, T, d_model]
        
        # 因果掩码
        causal_mask = self.generate_square_subsequent_mask(T, device)
        
        # Transformer Decoder
        output = self.decoder(
            tgt=token_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )
        
        # 输出投影
        logits = self.output_proj(output)  # [B, T, vocab_size]
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        device: str = "cpu",
    ) -> List[int]:
        """
        自回归生成 Action Token 序列
        
        Args:
            image: [1, 3, H, W] 输入图像
            max_tokens: 最大生成长度
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Nucleus 采样
        Returns:
            生成的 Token ID 列表
        """
        self.eval()
        
        # 初始 Token (空序列)
        generated = [PAD]
        
        for step in range(max_tokens):
            # 构建输入
            tokens = torch.tensor([generated], dtype=torch.long, device=device)
            
            # 前向传播
            logits = self.forward(image, tokens)  # [1, T, vocab_size]
            
            # 取最后一个位置的预测
            next_logits = logits[0, -1] / temperature  # [vocab_size]
            
            # Top-K 过滤
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, top_k)
                threshold = top_k_vals[-1]
                next_logits[next_logits < threshold] = float('-inf')
            
            # Top-P (Nucleus) 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[1:] = sorted_mask[:-1].clone()
                sorted_mask[0] = False
                indices_to_remove = sorted_mask.scatter(0, sorted_indices, sorted_mask)
                next_logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            generated.append(next_token)
            
            # 结束条件
            if next_token == END_SEQUENCE:
                break
        
        return generated[1:]  # 去掉初始 PAD


# 保留 List 导入
from typing import List
