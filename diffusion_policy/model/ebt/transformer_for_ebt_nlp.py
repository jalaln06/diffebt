# diffusion_policy/model/ebt/transformer_for_ebt_nlp.py

from typing import Optional, Tuple
import logging
import torch
import torch.nn as nn
import math
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute RoPE frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply rotary position embeddings."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class EBTAttention(nn.Module):
    """
    Split attention for EBT with dropout support.
    """
    def __init__(self, dim: int, n_heads: int, p_drop_attn: float = 0.1, causal_obs: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.causal_obs = causal_obs
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
        # Dropout
        self.attn_dropout = nn.Dropout(p_drop_attn)
        self.resid_dropout = nn.Dropout(p_drop_attn)
        
        # Initialize
        for layer in [self.wq, self.wk, self.wv, self.wo]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        n_obs_tokens: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, n_obs + n_action, D) - concatenated obs + action embeddings
            freqs_cis: Rotary position embeddings
            n_obs_tokens: Number of observation tokens
        """
        bsz, total_seqlen, _ = x.shape
        n_action_tokens = total_seqlen - n_obs_tokens
        
        # Project to Q, K, V
        xq = self.wq(x).view(bsz, total_seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, total_seqlen, self.n_heads, self.head_dim)
        xv = self.wv(x).view(bsz, total_seqlen, self.n_heads, self.head_dim)
        
        # Split into obs and action
        xq_obs = xq[:, :n_obs_tokens]
        xk_obs = xk[:, :n_obs_tokens]
        xv_obs = xv[:, :n_obs_tokens]
        
        xq_act = xq[:, n_obs_tokens:]
        xk_act = xk[:, n_obs_tokens:]
        xv_act = xv[:, n_obs_tokens:]
        
        # Apply RoPE
        xq_obs, xk_obs = apply_rotary_emb(xq_obs, xk_obs, freqs_cis[:n_obs_tokens])
        
        # Action tokens use positions [1:1+n_action_tokens] (shifted by 1)
        # This gives them position indices relative to obs tokens
        xq_act, xk_act = apply_rotary_emb(
            xq_act, 
            xk_act, 
            freqs_cis[n_obs_tokens : n_obs_tokens + n_action_tokens]  # FIXED: was freqs_cis[1:n_obs_tokens+1]
        )
        
        # ===== Part 1: Observation Self-Attention =====
        xq_obs = xq_obs.transpose(1, 2)
        xk_obs = xk_obs.transpose(1, 2)
        xv_obs = xv_obs.transpose(1, 2)
        
        scores_obs = torch.matmul(xq_obs, xk_obs.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if self.causal_obs:
            mask_obs = torch.triu(
                torch.ones(n_obs_tokens, n_obs_tokens, device=x.device),
                diagonal=1
            ) * float('-inf')
            scores_obs = scores_obs + mask_obs
        
        attn_obs = torch.softmax(scores_obs.float(), dim=-1).type_as(xq_obs)
        attn_obs = self.attn_dropout(attn_obs)  # Apply dropout
        output_obs = torch.matmul(attn_obs, xv_obs)
        output_obs = output_obs.transpose(1, 2).contiguous().view(bsz, n_obs_tokens, -1)
        
        # ===== Part 2: Action Attention (EBT Style) =====
        xq_act = xq_act.transpose(1, 2)
        xk_act = xk_act.transpose(1, 2)
        xv_act = xv_act.transpose(1, 2)
        
        # Actions attend to ALL observations
        scores_act_to_obs = torch.matmul(xq_act, xk_obs.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Add extra column for self-attention
        temp_col = torch.zeros(
            (bsz, self.n_heads, n_action_tokens, 1),
            dtype=scores_act_to_obs.dtype,
            device=scores_act_to_obs.device
        )
        scores_act = torch.cat([scores_act_to_obs, temp_col], dim=-1)
        
        # Compute self-attention scores (superdiagonal)
        self_attn_scores = (xq_act * xk_act).sum(dim=3) / math.sqrt(self.head_dim)
        
        # Insert self-attention on superdiagonal
        superdiag_rows = torch.arange(n_action_tokens)
        superdiag_cols = torch.full((n_action_tokens,), n_obs_tokens, dtype=torch.long)
        
        diagonal_mask = torch.zeros_like(scores_act)
        diagonal_mask[:, :, superdiag_rows, superdiag_cols] = self_attn_scores
        scores_act = scores_act + diagonal_mask
        
        # Softmax
        attn_act = torch.softmax(scores_act.float(), dim=-1).type_as(xq_act)
        attn_act = self.attn_dropout(attn_act)  # Apply dropout
        
        # Extract superdiagonal attention weights
        attn_act_self = attn_act[:, :, superdiag_rows, superdiag_cols].clone()
        
        # Remove self-attention column
        attn_act_to_obs = attn_act[:, :, :, :-1]
        
        # Compute output
        output_act_from_obs = torch.matmul(attn_act_to_obs, xv_obs)
        output_act_from_self = xv_act * attn_act_self.unsqueeze(-1)
        output_act = output_act_from_obs + output_act_from_self
        
        output_act = output_act.transpose(1, 2).contiguous().view(bsz, n_action_tokens, -1)
        
        # Concatenate outputs
        output = torch.cat([output_obs, output_act], dim=1)
        
        # Apply residual dropout
        return self.resid_dropout(self.wo(output))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, p_drop_attn: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(p_drop_attn)
        
        for layer in [self.w1, self.w2, self.w3]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(self, x):
        return self.dropout(self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int, p_drop_attn: float, causal_obs: bool):
        super().__init__()
        self.attention = EBTAttention(dim, n_heads, p_drop_attn, causal_obs)
        self.feed_forward = FeedForward(dim, hidden_dim, p_drop_attn)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, n_obs_tokens: int):
        h = x + self.attention(self.attention_norm(x), freqs_cis, n_obs_tokens)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class TransformerForEBT_NLP(ModuleAttrMixin):
    """
    EBT Transformer using NLP-style split attention.
    Adapted from EBTDefault for robotics control.
    
    Compatible with all diffusion_policy parameters.
    """
    def __init__(
        self,
        action_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 0,
        n_layer: int = 8,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = True,  # For action tokens (always used in EBT)
        obs_as_cond: bool = True,   # Obs always used as conditioning in this architecture
        n_cond_layers: int = 0,     # Not used in NLP-style, kept for compatibility
        energy_head_hidden_dim: int = None,
        max_seq_len: int = 512,
    ):
        super().__init__()
        
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        assert obs_as_cond and cond_dim > 0, "NLP-style EBT requires observation conditioning"
        
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_emb = n_emb
        self.obs_as_cond = obs_as_cond
        
        # Embeddings with dropout
        self.obs_emb = nn.Linear(cond_dim, n_emb)
        self.action_emb = nn.Linear(action_dim, n_emb)
        self.drop = nn.Dropout(p_drop_emb)
        
        # Transformer layers
        hidden_dim = 4 * n_emb
        self.layers = nn.ModuleList([
            TransformerBlock(n_emb, n_head, hidden_dim, p_drop_attn, causal_obs=False)
            for _ in range(n_layer)
        ])
        
        self.norm = RMSNorm(n_emb)
        
        # Energy head with optional hidden layer
        if energy_head_hidden_dim is None:
            energy_head_hidden_dim = n_emb
        
        if energy_head_hidden_dim == n_emb:
            # Simple linear layer
            self.energy_head = nn.Linear(n_emb, 1, bias=False)
        else:
            # MLP head
            self.energy_head = nn.Sequential(
                nn.Linear(n_emb, energy_head_hidden_dim),
                nn.Mish(),
                nn.Linear(energy_head_hidden_dim, 1)
            )
        
        # RoPE frequencies
        head_dim = n_emb // n_head
        self.freqs_cis = precompute_freqs_cis(head_dim, max_seq_len)
        
        # Initialize weights
        nn.init.normal_(self.obs_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.action_emb.weight, mean=0.0, std=0.02)
        if isinstance(self.energy_head, nn.Linear):
            nn.init.normal_(self.energy_head.weight, mean=0.0, std=0.02)
        else:
            for module in self.energy_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        logger.info(
            f"EBT NLP Transformer - "
            f"action_dim: {action_dim}, "
            f"horizon: {horizon}, "
            f"n_obs_steps: {n_obs_steps}, "
            f"n_layer: {n_layer}, "
            f"n_head: {n_head}, "
            f"n_emb: {n_emb}, "
            f"params: {sum(p.numel() for p in self.parameters()):e}"
        )

    def forward(
        self,
        actions: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            actions: (B, horizon, action_dim)
            cond: (B, n_obs_steps, cond_dim) - observation features
        
        Returns:
            energy: (B,) - scalar energy
        """
        bsz = actions.shape[0]
        
        # Embed observations and actions
        obs_emb = self.obs_emb(cond)  # (B, n_obs_steps, n_emb)
        action_emb = self.action_emb(actions)  # (B, horizon, n_emb)
        
        # Apply dropout
        obs_emb = self.drop(obs_emb)
        action_emb = self.drop(action_emb)
        
        # Concatenate [obs, actions] - NLP-style
        x = torch.cat([obs_emb, action_emb], dim=1)  # (B, n_obs+horizon, n_emb)
        
        # Get RoPE frequencies
        seq_len = x.shape[1]
        self.freqs_cis = self.freqs_cis.to(x.device)
        freqs_cis = self.freqs_cis[:seq_len]
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, freqs_cis, self.n_obs_steps)
        
        x = self.norm(x)
        
        # Extract action portion only
        action_features = x[:, self.n_obs_steps:]  # (B, horizon, n_emb)
        
        # Predict energy
        if isinstance(self.energy_head, nn.Linear):
            energies = self.energy_head(action_features)  # (B, horizon, 1)
        else:
            energies = self.energy_head(action_features)  # (B, horizon, 1)
        
        # Pool to scalar energy (mean pooling over horizon)
        energy = energies.mean(dim=1).squeeze(-1)  # (B,)
        
        return energy

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """Parameter groups for optimizer."""
        decay = set()
        no_decay = set()
        
        whitelist = (nn.Linear,)
        blacklist = (RMSNorm,)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)
        
        # Add dummy variable to no_decay in case it exists
        no_decay.add("_dummy_variable")
        
        # Validate
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        
        assert len(inter_params) == 0, \
            f"Parameters in both decay/no_decay: {inter_params}"
        
        # Handle remaining parameters
        remaining = param_dict.keys() - union_params
        if len(remaining) > 0:
            logger.warning(f"Adding remaining params to no_decay: {remaining}")
            no_decay.update(remaining)
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return optim_groups