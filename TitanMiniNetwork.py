# TitanMiniNetwork.py (stabilized)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1) Positional / Relative Bias (bounded)
# ==============================================================================

class RelativePositionBias(nn.Module):
    """
    2D (rank,file) relative bias with bounding to avoid runaway scores.
    """
    def __init__(self, num_heads, max_relative_position=7, bias_scale=2.0):
        super().__init__()
        self.num_heads = num_heads
        self.max_relative_position = max_relative_position
        self.bias_scale = float(bias_scale)  # scale after tanh bound
        num_buckets = 2 * max_relative_position + 1
        # [rank_bucket, file_bucket, heads]
        self.relative_bias_table = nn.Parameter(
            torch.zeros(num_buckets, num_buckets, num_heads)
        )

    def forward(self, seq_len: int):
        assert seq_len in (64, 65)
        board_len = 64
        device = self.relative_bias_table.device
        positions = torch.arange(board_len, device=device)
        ranks = positions // 8
        files = positions % 8
        dr = ranks.unsqueeze(1) - ranks.unsqueeze(0)
        df = files.unsqueeze(1) - files.unsqueeze(0)
        r_idx = dr.clamp(-self.max_relative_position, self.max_relative_position) + self.max_relative_position
        f_idx = df.clamp(-self.max_relative_position, self.max_relative_position) + self.max_relative_position
        # Bound via tanh to keep within [-bias_scale, bias_scale]
        bias = torch.tanh(self.relative_bias_table[r_idx, f_idx]) * self.bias_scale  # [64,64,heads]
        bias_64 = bias.permute(2, 0, 1).unsqueeze(0)  # [1,H,64,64]

        if seq_len == 64:
            return bias_64

        # For [CLS]+64: pad the [0,0] position with 0 and place bias in 1:,1:
        out = torch.zeros(1, self.num_heads, 65, 65, device=device, dtype=bias_64.dtype)
        out[:, :, 1:, 1:] = bias_64
        return out


class ChessPositionalEncoding(nn.Module):
    """
    Rich positional encoding: absolute + file/rank/diag/anti-diag.
    """
    def __init__(self, d_model, max_seq_len=64):
        super().__init__()
        self.absolute_pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.file_embedding = nn.Embedding(8, d_model)
        self.rank_embedding = nn.Embedding(8, d_model)
        self.diag_embedding = nn.Embedding(15, d_model)
        self.anti_diag_embedding = nn.Embedding(15, d_model)

    def forward(self, x):
        seq_len = x.shape[1]
        assert seq_len == 64, "ChessPositionalEncoding expects 64 tokens (8x8)."
        device = x.device
        positions = torch.arange(seq_len, device=device)
        files = positions % 8
        ranks = positions // 8
        diagonals = ranks + files
        anti_diagonals = ranks - files + 7
        file_emb = self.file_embedding(files)
        rank_emb = self.rank_embedding(ranks)
        diag_emb = self.diag_embedding(diagonals)
        anti_diag_emb = self.anti_diag_embedding(anti_diagonals)
        return (self.absolute_pos_embedding[:, :seq_len, :]
                + file_emb + rank_emb + diag_emb + anti_diag_emb)

# ==============================================================================
# 2) Transformer Core (attention computed in fp32 for stability)
# ==============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    MHA with fp32 attention math (under AMP) for numerical stability.
    """
    def __init__(self, d_model, num_heads, dropout=0.1, attn_clamp: float = 50.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relative_position_bias = RelativePositionBias(num_heads)
        self.attn_clamp = float(attn_clamp)  # optional pre-softmax clamp

    def forward(self, x, mask=None):
        """
        x: [B, T, D]
        mask: optional boolean or 0/1 mask broadcastable to [B, 1, T, T]
        """
        B, T, _ = x.shape
        orig_dtype = x.dtype

        # Disable autocast for attention math; do it in fp32
        with torch.cuda.amp.autocast(enabled=False):
            xf = x.float()
            Q = self.W_q(xf).view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # [B,H,T,d]
            K = self.W_k(xf).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(xf).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B,H,T,T]

            # Relative bias (fp32)
            if T in (64, 65):
                scores = scores + self.relative_position_bias(T)

            if mask is not None:
                # Expect mask broadcastable to [B,1,T,T] with 1=keep, 0=mask
                # Use -1e4 (fp32-safe, fp16-safe) to avoid -inf underflow
                scores = scores.masked_fill(mask == 0, -1e4)

            # Optional clamp to prevent extreme logits
            if self.attn_clamp is not None and self.attn_clamp > 0:
                scores = scores.clamp(min=-self.attn_clamp, max=self.attn_clamp)

            # Softmax in fp32
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            context = torch.matmul(attn, V)  # [B,H,T,d]
            context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)

            out = self.W_o(context).to(orig_dtype)

        return out

class GEGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.proj = nn.Linear(d_model, d_ff * 2)
        self.out = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return self.out(F.gelu(x1) * x2)

class TransformerBlock(nn.Module):
    """
    Pre-LN block; attention fp32; FFN GEGLU.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = GEGLU(d_model, d_ff)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        return x

# ==============================================================================
# 3) Output Heads
# ==============================================================================

class PolicyHead(nn.Module):
    """
    Outputs logits in PLANE-MAJOR layout to match encoder/mask/targets:
      index = plane * 64 + square
    Given board_features [B, 64, D], we first project to [B, 64, 72] (per-square, per-plane),
    then permute to [B, 72, 64] and flatten -> [B, 4608].
    """
    def __init__(self, d_model, num_move_actions_per_square=72, logit_scale=1.0):
        super().__init__()
        self.num_move_actions_per_square = num_move_actions_per_square  # 72
        self.proj = nn.Linear(d_model, num_move_actions_per_square)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale)))

    def forward(self, x):
        # x: [B, 64, D]
        logits_sq_plane = self.proj(x) * self.logit_scale      # [B, 64, 72]
        logits_plane_sq = logits_sq_plane.permute(0, 2, 1).contiguous()  # [B, 72, 64]
        logits = logits_plane_sq.view(x.shape[0], -1)          # [B, 4608] PLANE-MAJOR
        return logits

class ValueHead(nn.Module):
    def __init__(self, d_model, legacy_mode=False):
        super().__init__()
        self.legacy_mode = legacy_mode
        self.d_model = d_model
        if legacy_mode:
            self.fc1 = nn.Linear(d_model, d_model // 2); self.activation1 = nn.GELU(); self.dropout1 = nn.Dropout(0.1)
            self.fc2 = nn.Linear(d_model // 2, 1); self.output_activation = nn.Tanh()
        else:
            self.fc1 = nn.Linear(d_model, d_model // 2); self.activation1 = nn.GELU(); self.dropout1 = nn.Dropout(0.1)
            self.fc2 = nn.Linear(d_model // 2, 128); self.activation2 = nn.GELU(); self.dropout2 = nn.Dropout(0.1)
            self.fc3 = nn.Linear(128, 1); self.output_activation = nn.Tanh()
    def forward(self, x):
        x = x.squeeze(1)
        if self.legacy_mode:
            x = self.fc1(x); x = self.activation1(x); x = self.dropout1(x)
            x = self.fc2(x); x = self.output_activation(x)
        else:
            x = self.fc1(x); x = self.activation1(x); x = self.dropout1(x)
            x = self.fc2(x); x = self.activation2(x); x = self.dropout2(x)
            x = self.fc3(x); x = self.output_activation(x)
        return x

class WDLValueHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.value_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_model // 2, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 3),
        )
    def forward(self, x):
        return self.value_proj(x.squeeze(1))

# ==============================================================================
# 4) Titan-Mini (stabilized)
# ==============================================================================

class TitanMini(nn.Module):
    def __init__(
        self,
        num_layers=13, d_model=512, num_heads=8, d_ff=1920, dropout=0.1,
        policy_weight=1.0, input_planes=16, use_wdl=True, legacy_value_head=False,
    ):
        super().__init__()
        self.policy_weight = policy_weight
        self.d_model = d_model
        self.use_wdl = use_wdl
        self.input_planes = input_planes

        # Sparse input path (16 planes)
        if input_planes < 12:
            raise ValueError(f"Input planes must be >=12, got {input_planes}")
        self.piece_type_embedding = nn.Embedding(7, d_model)  # Empty=0 + 6
        self.color_embedding = nn.Embedding(2, d_model)       # 0=friendly/empty, 1=enemy
        num_global_features = max(0, input_planes - 12)
        self.global_feature_proj = nn.Linear(num_global_features or 1, d_model) if num_global_features > 0 else None

        self.positional_encoding = ChessPositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        # NEW: input norm to stabilize first pass
        self.input_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.output_norm = nn.LayerNorm(d_model, eps=1e-5)

        self.value_head = WDLValueHead(d_model) if use_wdl else ValueHead(d_model, legacy_mode=legacy_value_head)
        self.policy_head = PolicyHead(d_model, num_move_actions_per_square=72, logit_scale=1.0)

        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self._init_parameters()
        self._init_head_specific_weights()

    # ---- sparse input helpers ----
    def _process_sparse_input(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W)
        friendly_planes = x_flat[:, 0:12:2]
        enemy_planes = x_flat[:, 1:12:2]

        is_friendly = torch.sum(friendly_planes, dim=1) > 0.5
        is_enemy = torch.sum(enemy_planes, dim=1) > 0.5
        color_indices = is_enemy.long()

        combined_planes = friendly_planes + enemy_planes
        type_indices_0_5 = torch.argmax(combined_planes, dim=1)
        is_occupied = (is_friendly + is_enemy) > 0.5
        piece_type_indices = (type_indices_0_5 + 1) * is_occupied.long()

        type_embs = self.piece_type_embedding(piece_type_indices)  # [B,64,D]
        color_embs = self.color_embedding(color_indices)           # [B,64,D]
        board_tokens = type_embs + color_embs  # [B,64,D]

        return board_tokens, type_embs, is_friendly.float(), is_enemy.float()

    # ---- init ----
    def _init_parameters(self):
        std = 0.02
        def trunc_normal_(t, s):
            if t is None: return
            if hasattr(nn.init, "trunc_normal_"):
                if t.dtype in (torch.float16, torch.bfloat16):
                    t.data = t.data.to(torch.float32)
                nn.init.trunc_normal_(t, std=s, a=-2*s, b=2*s)
            else:
                nn.init.normal_(t, std=s)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Embedding)):
                trunc_normal_(getattr(m, 'weight', None), std)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None: nn.init.constant_(m.weight, 1.0)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, RelativePositionBias):
                trunc_normal_(m.relative_bias_table, std)

        if hasattr(self.positional_encoding, 'absolute_pos_embedding'):
            trunc_normal_(self.positional_encoding.absolute_pos_embedding, std)

    def _init_head_specific_weights(self):
        std_head = 0.01
        def trunc_normal_(t, s):
            if t is None: return
            if hasattr(nn.init, "trunc_normal_"):
                if t.dtype in (torch.float16, torch.bfloat16):
                    t.data = t.data.to(torch.float32)
                nn.init.trunc_normal_(t, std=s, a=-2*s, b=2*s)
            else:
                nn.init.normal_(t, std=s)

        if hasattr(self.policy_head, 'proj'):
            trunc_normal_(self.policy_head.proj.weight, std_head)
        # Final value layers
        if self.use_wdl:
            final_layer = self.value_head.value_proj[-1]
            if isinstance(final_layer, nn.Linear):
                trunc_normal_(final_layer.weight, std_head)
        else:
            final_layer = self.value_head.fc2 if self.value_head.legacy_mode else self.value_head.fc3
            trunc_normal_(final_layer.weight, std_head)

    # ---- forward ----
    def forward(self, x, value_target=None, policy_target=None, policy_mask=None,
                valueTarget=None, policyTarget=None, policyMask=None):
        # Backward-compat kwargs
        if valueTarget is not None: value_target = valueTarget
        if policyTarget is not None: policy_target = policyTarget
        if policyMask is not None: policy_mask = policyMask

        B = x.shape[0]
        # Sparse input path
        tokens_symmetric, type_embs, is_friendly, is_enemy = self._process_sparse_input(x)

        # Global context (castling rights etc.)
        global_proj = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
        if self.global_feature_proj is not None:
            global_features = x[:, 12:, 0, 0]  # [B, num_global]
            # ensure non-empty last dim
            global_proj = self.global_feature_proj(global_features)

        # Signed material sum for CLS (compute in fp32 for stability)
        calc_dtype = torch.float32 if x.dtype == torch.float16 else x.dtype
        sign_mask = (is_friendly.to(calc_dtype) - is_enemy.to(calc_dtype)).unsqueeze(-1)  # [B,64,1]
        signed_material_sum = torch.sum(type_embs.to(calc_dtype) * sign_mask, dim=1)      # [B,D]
        cls = (signed_material_sum.to(x.dtype) + global_proj).unsqueeze(1)                 # [B,1,D]

        # Enhance board tokens with global context
        board_tokens = tokens_symmetric + global_proj.unsqueeze(1)  # [B,64,D]

        # Positional encodings (+ small scale multiplier helps)
        pos = self.positional_encoding(board_tokens)
        board_tokens = board_tokens + pos

        # Combine, then input LayerNorm for extra stability
        x_combined = torch.cat([cls, board_tokens], dim=1)  # [B,65,D]
        x_combined = self.input_norm(x_combined)

        # Transformer stack
        for block in self.transformer_blocks:
            x_combined = block(x_combined, mask=None)

        x_combined = self.output_norm(x_combined)
        cls_features = x_combined[:, 0:1, :]
        board_features = x_combined[:, 1:, :]

        value = self.value_head(cls_features)        # [B,1] or [B,3]
        policy = self.policy_head(board_features)    # [B,4608]

        # ---- TRAINING PATH ----
        if self.training:
            assert value_target is not None and policy_target is not None

            # Apply legal move mask to policy logits (fp16-safe constant)
            if policy_mask is not None:
                pm = policy_mask.view(B, -1)
                policy = policy.masked_fill(pm == 0, -1e4)

            # Guard against non-finite logits
            if not torch.isfinite(policy).all():
                policy = torch.nan_to_num(policy, nan=0.0, posinf=1e4, neginf=-1e4)

            # ----- Value loss -----
            if self.use_wdl:
                # value_target can be [B,3] (WDL) or [B,1] scalar in [-1,1]
                if value_target.dim() == 2 and value_target.size(1) == 3:
                    wdl_targets = value_target
                else:
                    v = value_target.view(-1)
                    W = torch.relu(v)
                    L = torch.relu(-v)
                    D = torch.clamp(1.0 - W - L, min=0.0)
                    wdl_targets = torch.stack([W, D, L], dim=1)
                log_probs = F.log_softmax(value.float(), dim=1)
                wdl_targets = wdl_targets / (wdl_targets.sum(dim=1, keepdim=True) + 1e-8)
                value_loss = -(wdl_targets * log_probs).sum(dim=1).mean()
            else:
                value_loss = self.mse_loss(value, value_target)

            # ----- Policy loss -----
            if policy_target.dim() == 1 or (policy_target.dim() == 2 and policy_target.size(1) == 1):
                # hard targets (indices)
                policy_loss = self.cross_entropy_loss(policy, policy_target.view(B))
            else:
                # soft targets (distributions)
                pt = policy_target.view(B, -1).float()
                pt = pt + 1e-8
                pt = pt / pt.sum(dim=1, keepdim=True)
                logp = F.log_softmax(policy.float(), dim=1)  # compute in fp32
                policy_loss = -(pt * logp).sum(dim=1).mean()

            total_loss = value_loss + self.policy_weight * policy_loss
            # Final finite guard
            if not torch.isfinite(total_loss):
                # zero out bad batch to avoid poisoning optimizer
                total_loss = torch.zeros((), device=x.device, dtype=value.dtype, requires_grad=True)
                value_loss = total_loss
                policy_loss = total_loss

            return total_loss, value_loss, policy_loss

        # ---- INFERENCE PATH ----
        else:
            if policy_mask is not None:
                pm = policy_mask.view(B, -1)
                policy = policy.masked_fill(pm == 0, -1e4)
            policy_softmax = F.softmax(policy, dim=1)

            if self.use_wdl:
                wdl = F.softmax(value, dim=1)
                value_scalar = wdl[:, 0:1] - wdl[:, 2:3]
                return value_scalar, policy_softmax
            else:
                return value, policy_softmax


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
