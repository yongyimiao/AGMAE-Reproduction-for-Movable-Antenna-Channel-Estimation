import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import Block
from util.pos_embed import get_2d_sincos_pos_embed


# ---------------------------------------------------------------------------
# Graph helpers  (直接复用 models_mae_fas_GNN.py 中的实现)
# ---------------------------------------------------------------------------

def build_knn_edge_index(num_nodes: int, grid_size: int, k: int,
                         device: torch.device) -> torch.Tensor:
    """
    为 (grid_size x grid_size) 的网格节点构建静态 k-NN 边索引。
    返回 edge_index: (2, N*k)，整个训练过程复用，无需每 batch 重建。
    """
    rows = torch.div(torch.arange(num_nodes, device=device),
                     grid_size, rounding_mode='floor')
    cols = torch.arange(num_nodes, device=device) % grid_size
    coords = torch.stack([rows, cols], dim=1).float()       # (N, 2)

    diff = coords.unsqueeze(1) - coords.unsqueeze(0)        # (N, N, 2)
    dist = (diff ** 2).sum(-1)                              # (N, N)
    dist.fill_diagonal_(float('inf'))

    _, nn_idx = dist.topk(k, dim=1, largest=False)          # (N, k)
    src = (torch.arange(num_nodes, device=device)
           .unsqueeze(1).expand_as(nn_idx).reshape(-1))
    dst = nn_idx.reshape(-1)
    return torch.stack([src, dst], dim=0)                   # (2, N*k)


# ---------------------------------------------------------------------------
# GAT layer  (论文 eq.11，纯 PyTorch，无 DGL 依赖)
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """单头 Graph Attention Layer，特征维度 F_dim 保持不变。"""

    def __init__(self, F_dim: int):
        super().__init__()
        self.Wa    = nn.Linear(F_dim, F_dim, bias=False)
        self.We    = nn.Linear(2 * F_dim, 1, bias=False)
        self.act   = nn.LeakyReLU(negative_slope=0.2)
        self.sigma = nn.GELU()

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """h: (B, N, F)  ->  (B, N, F)"""
        B, N, F_size = h.shape
        src, dst = edge_index[0], edge_index[1]             # (E,)

        h_src = h[:, src, :]                                # (B, E, F)
        h_dst = h[:, dst, :]                                # (B, E, F)

        # e(h_i, h_j) = a · LReLU(We · [h_i || h_j])
        e = self.act(
            self.We(torch.cat([h_dst, h_src], dim=-1))
        ).squeeze(-1)                                       # (B, E)

        # softmax per destination node
        alpha = torch.full((B, N, N), float('-inf'),
                           device=h.device, dtype=e.dtype)
        alpha[:, dst, src] = e
        alpha = F.softmax(alpha, dim=-1)                    # (B, N, N)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # h'_i = sigma( sum_j alpha_ij * Wa * h_j )
        agg = torch.bmm(alpha, self.Wa(h))                  # (B, N, F)
        return self.sigma(agg)


class GATBlock(nn.Module):
    """
    AGMAE decoder block (eq.10):
    GAT + residual + LayerNorm + FFN + residual
    """

    def __init__(self, F_dim: int, mlp_ratio: float = 4.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(F_dim)
        self.gat   = GATLayer(F_dim)
        self.norm2 = norm_layer(F_dim)
        hidden = int(F_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(F_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, F_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = h + self.gat(self.norm1(h), edge_index)
        h = h + self.ffn(self.norm2(h))
        return h


# ---------------------------------------------------------------------------
# Patch Embedding（与原始 noise 版本相同）
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """1x1 Patch Embedding for per-port CSI"""

    def __init__(self, img_size=20, patch_size=1, in_chans=2, embed_dim=256):
        super().__init__()
        self.img_size    = img_size
        self.grid_size   = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.proj(x)                     # [B, embed_dim, H, W]
        x = x.flatten(2).transpose(1, 2)     # [B, num_patches, embed_dim]
        return x


# ---------------------------------------------------------------------------
# 主模型：保留双输入 (noisy / clean)，decoder 换成 GAT
# ---------------------------------------------------------------------------

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=20, patch_size=1, in_chans=2,
                 embed_dim=256, depth=12, num_heads=4,
                 decoder_embed_dim=128, decoder_depth=12, decoder_num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 gat_k_neighbors: int = 8):
        super().__init__()

        # ---- Encoder（不变）--------------------------------------------
        self.patch_embed  = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches       = self.patch_embed.num_patches
        self.grid_size    = self.patch_embed.grid_size

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # ---- Decoder：GAT 替换 Transformer Block -----------------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            GATBlock(decoder_embed_dim, mlp_ratio=mlp_ratio,
                     norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        # ---- GAT 拓扑（按 device 缓存，懒构建）------------------------
        self.gat_k_neighbors    = gat_k_neighbors
        self._edge_index_cache: dict = {}

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    # ------------------------------------------------------------------
    # 权重初始化
    # ------------------------------------------------------------------
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5), cls_token=False)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    # GAT 边索引（按 device 缓存）
    # ------------------------------------------------------------------
    def _get_edge_index(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._edge_index_cache:
            self._edge_index_cache[key] = build_knn_edge_index(
                self.patch_embed.num_patches,
                self.grid_size,
                self.gat_k_neighbors,
                device)
        return self._edge_index_cache[key]

    # ------------------------------------------------------------------
    # patchify / unpatchify / random_masking（与原版相同）
    # ------------------------------------------------------------------
    def patchify(self, imgs):
        """imgs: (N, 2, H, W) -> (N, H*W, 2)"""
        return imgs.reshape(imgs.shape[0], 2, -1).permute(0, 2, 1)

    def unpatchify(self, x):
        """x: (N, 400, 2) -> (N, 2, 20, 20)"""
        grid_size = self.patch_embed.grid_size
        return x.permute(0, 2, 1).reshape(x.shape[0], 2, grid_size, grid_size)

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(round(L * (1 - mask_ratio)))
        len_keep = max(1, min(L - 1, len_keep))
        noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore  = torch.argsort(ids_shuffle, dim=1)
        ids_keep     = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    # ------------------------------------------------------------------
    # Encoder：以有噪 CSI 作为输入
    # ------------------------------------------------------------------
    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    # ------------------------------------------------------------------
    # Decoder：GAT 局部扩散恢复全部端口
    # ------------------------------------------------------------------
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)                                    # (B, Nb, F)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)       # (B, Na, F)
        x_ = torch.cat([x, mask_tokens], dim=1)                     # (B, Ns, F)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x = x_ + self.decoder_pos_embed                              # (B, Ns, F)

        edge_index = self._get_edge_index(x.device)
        for blk in self.decoder_blocks:
            x = blk(x, edge_index)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)                                     # (B, Ns, in_chans)
        return x

    # ------------------------------------------------------------------
    # Loss（保留原 noise 版本的双损失设计，以 clean CSI 为目标）
    # ------------------------------------------------------------------
    def forward_loss_MSE(self, imgs_clean, pred, mask):
        """仅掩码区域的 MSE，可用于调试对比。"""
        target = self.patchify(imgs_clean)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward_loss(self, imgs_clean, pred, mask, alpha=0.5):
        """
        NMSE 损失，目标为无噪 clean CSI。
        loss = alpha * masked_nmse + (1-alpha) * unmasked_nmse
        返回: (total_loss, masked_loss_scalar, unmasked_loss_scalar)
        """
        target = self.patchify(imgs_clean)                           # (B, L, 2)

        mse   = ((pred - target) ** 2).mean(dim=-1)                  # (B, L)
        power = (target ** 2).mean(dim=-1) + 1e-6
        nmse  = mse / power                                          # (B, L)

        masked_loss   = (nmse * mask).sum() / mask.sum()
        unmasked_loss = (nmse * (1 - mask)).sum() / (1 - mask).sum()

        loss = alpha * masked_loss + (1 - alpha) * unmasked_loss
        return loss, masked_loss.item(), unmasked_loss.item()

    # ------------------------------------------------------------------
    # Forward：encoder 吃有噪数据，loss 对比无噪 clean
    # ------------------------------------------------------------------
    def forward(self, imgs_noisy, imgs_clean, mask_ratio=0.84, alpha=0.5):
        latent, mask, ids_restore = self.forward_encoder(imgs_noisy, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss, masked_loss, unmasked_loss = self.forward_loss(
            imgs_clean, pred, mask, alpha=alpha)
        return loss, pred, mask, masked_loss, unmasked_loss


# ---------------------------------------------------------------------------
# 模型工厂函数
# ---------------------------------------------------------------------------

def mae_fas_channel_model(**kwargs):
    return MaskedAutoencoderViT(
        img_size=20, patch_size=1, in_chans=2,
        embed_dim=256, depth=12, num_heads=4,
        decoder_embed_dim=128, decoder_depth=12, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

def mae_fas_channel_modelv2(**kwargs):
    return MaskedAutoencoderViT(
        img_size=10, patch_size=1, in_chans=2,
        embed_dim=256, depth=12, num_heads=8,
        decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

def mae_fas_channel_modelv3(**kwargs):
    return MaskedAutoencoderViT(
        img_size=20, patch_size=1, in_chans=2,
        embed_dim=256, depth=16, num_heads=8,
        decoder_embed_dim=128, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

def mae_fas_channel_model_pos(**kwargs):
    return MaskedAutoencoderViT(
        img_size=20, patch_size=1, in_chans=2,
        embed_dim=288, depth=12, num_heads=4,
        decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )


mae_fas_channel_model     = mae_fas_channel_model
mae_fas_channel_modelv2   = mae_fas_channel_modelv2
mae_fas_channel_modelv3   = mae_fas_channel_modelv3
mae_fas_channel_model_pos = mae_fas_channel_model_pos
