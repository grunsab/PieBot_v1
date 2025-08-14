# TitanMiniNetwork.py (fixed)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1) Positional encoding and relative bias
# ==============================================================================

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_relative_position=7):
        super().__init__()
        self.num_heads = num_heads
        self.max_relative_position = max_relative_position
        num_buckets = 2 * max_relative_position + 1
        self.relative_bias_table = nn.Parameter(
            torch.zeros(num_buckets, num_buckets, num_heads)
        )

    def forward(self, seq_len: int):
        # supports 64 tokens (board) or 65 (CLS + board)
        assert seq_len in (64, 65)
        board_len = 64
        device = self.relative_bias_table.device

        positions = torch.arange(board_len, device=device)
        ranks = positions // 8
        files = positions % 8

        dr = ranks.unsqueeze(1) - ranks.unsqueeze(0)
        df = files.unsqueeze(1) - files.unsqueeze(0)

        dr_idx = dr.clamp(-self.max_relative_position, self.max_relative_position) + self.max_relative_position
        df_idx = df.clamp(-self.max_relative_position, self.max_relative_position) + self.max_relative_position

        bias = self.relative_bias_table[dr_idx, df_idx]  # [64,64,H]
        bias_64 = bias.permute(2, 0, 1).unsqueeze(0)     # [1,H,64,64]

        if seq_len == 64:
            return bias_64

        out = torch.zeros(1, self.num_heads, 65, 65, device=device, dtype=bias_64.dtype)
        out[:, :, 1:, 1:] = bias_64  # no bias involving CLS
        return out


class ChessPositionalEncoding(nn.Module):
    """
    Rich positional encoding for 8x8 boards: absolute + file/rank + diagonals.
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
        diags = ranks + files
        anti = ranks - files + 7

        pe = (
            self.absolute_pos_embedding[:, :seq_len, :]
            + self.file_embedding(files)
            + self.rank_embedding(ranks)
            + self.diag_embedding(diags)
            + self.anti_diag_embedding(anti)
        )
        return pe


# ==============================================================================
# 2) Transformer core
# ==============================================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
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

    def forward(self, x, mask=None):
        B, T, D = x.shape
        H = self.num_heads
        q = self.W_q(x).view(B, T, H, self.d_k).transpose(1, 2)  # [B,H,T,d]
        k = self.W_k(x).view(B, T, H, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B,H,T,T]
        if T in (64, 65):
            scores = scores + self.relative_position_bias(T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4 if scores.dtype == torch.float16 else -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v)  # [B,H,T,d]
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(ctx)


class GEGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.proj = nn.Linear(d_model, 2 * d_ff)
        self.out = nn.Linear(d_ff, d_model)

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)
        return self.out(F.gelu(a) * b)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = GEGLU(d_model, d_ff)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop1(self.attn(self.norm1(x), mask))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


# ==============================================================================
# 3) Output heads
# ==============================================================================

class PolicyHead(nn.Module):
    """
    Projects per-square features to 72 directional planes and flattens **plane-major**.
    This matches the rest of the codebase: index = plane*64 + square.
    """
    def __init__(self, d_model, num_move_actions_per_square=72):
        super().__init__()
        self.num_move_actions_per_square = num_move_actions_per_square
        self.proj = nn.Linear(d_model, num_move_actions_per_square)

    def forward(self, x):  # x: [B,64,d_model]
        logits_sq_plane = self.proj(x)                      # [B,64,72]
        logits_plane_sq = logits_sq_plane.permute(0, 2, 1)  # [B,72,64]  <-- plane-major
        return logits_plane_sq.contiguous().view(x.size(0), -1)  # [B,72*64=4608]


class DualValueHead(nn.Module):
    """
    Produces both a scalar [-1,1] value and WDL logits from a shared trunk.
    """
    def __init__(self, d_model, hidden=256, dropout=0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.scalar_out = nn.Linear(hidden, 1)
        self.wdl_out = nn.Linear(hidden, 3)

    def forward(self, x):  # x: [B,1,d_model] or [B,d_model]
        x = x.squeeze(1)
        h = self.trunk(x)
        scalar = torch.tanh(self.scalar_out(h))     # [-1,1]
        wdl_logits = self.wdl_out(h)                # unnormalized
        return scalar, wdl_logits


# ==============================================================================
# 4) TitanMini
# ==============================================================================

class TitanMini(nn.Module):
    def __init__(
        self,
        num_layers=13, d_model=512, num_heads=8, d_ff=1920, dropout=0.1,
        policy_weight=1.0, input_planes=112, use_wdl=True, legacy_value_head=False,
        # light-touch aux losses (do not require caller changes)
        wdl_weight=0.5, material_weight=0.05, calibration_weight=0.25, material_scale_cp=600.0,
        mask_illegal_in_training=True,
    ):
        super().__init__()

        self.policy_weight = policy_weight
        self.d_model = d_model
        self.use_wdl = use_wdl
        self.input_planes = input_planes

        # Aux loss weights
        self.wdl_weight = float(wdl_weight)
        self.scalar_weight = 1.0 - self.wdl_weight
        self.material_weight = float(material_weight)
        self.calibration_weight = float(calibration_weight)
        self.material_scale_cp = float(material_scale_cp)
        self.mask_illegal_in_training = bool(mask_illegal_in_training)

        # Decide sparse vs dense input
        self.use_sparse = (input_planes <= 16)

        # ====== Input modules ======
        if self.use_sparse:
            if input_planes < 12:
                raise ValueError(f"Sparse mode expects at least 12 planes, got {input_planes}")
            # piece planes: friendly indices [0,2,4,6,8,10], enemy [1,3,5,7,9,11]
            # Order (from encoder): [P, R, B, N, Q, K]
            self.piece_type_embedding = nn.Embedding(6, d_model)     # no "empty" inside this table
            self.color_embedding = nn.Embedding(2, d_model)          # 0=friendly, 1=enemy
            self.empty_embedding = nn.Parameter(torch.zeros(1, d_model))

            num_global_features = max(0, input_planes - 12)          # castling rights etc.
            self.global_feature_proj = nn.Linear(num_global_features, d_model) if num_global_features > 0 else None

            # Learnable CLS (big difference for value stability)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

            # Fixed piece values (centipawns) matching plane order [P, R, B, N, Q, K]
            cp = torch.tensor([100.0, 500.0, 300.0, 300.0, 900.0, 0.0], dtype=torch.float32)
            self.register_buffer("piece_values_cp", cp, persistent=False)

            # Project scalar material summary into d_model
            self.material_proj = nn.Sequential(
                nn.Linear(1, d_model // 2), nn.GELU(),
                nn.Linear(d_model // 2, d_model)
            )
        else:
            # Dense path (e.g., 112 enhanced planes)
            self.input_projection = nn.Conv2d(input_planes, d_model, kernel_size=1)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.global_feature_proj = None
            # No need for fixed values; dense features can carry them

        self.positional_encoding = ChessPositionalEncoding(d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(d_model)

        # Value + Policy heads
        self.value_head = DualValueHead(d_model)
        self.policy_head = PolicyHead(d_model, num_move_actions_per_square=72)

        # Losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # Init
        self._init_parameters()
        self._init_head_specific_weights()

    # ---------- Sparse input processing ----------
    def _process_sparse_input(self, x):
        """
        x: [B,C,8,8] with classic 16 planes
        Returns:
          board_tokens: [B,64,d_model]
          cls_seed:     [B,1,d_model]  (learnable cls + material/global)
          material_proxy_scalar: [B,1]  (tanh(cp/scale)) used only for a tiny aux loss
        """
        B, C, H, W = x.shape
        assert H == 8 and W == 8
        flat = x.view(B, C, H * W)  # [B,C,64]

        friendly = flat[:, 0:12:2, :]  # [B,6,64] order [P,R,B,N,Q,K]
        enemy    = flat[:, 1:12:2, :]  # [B,6,64]

        # occupancy by side
        occ_f = (friendly.sum(dim=1) > 0.5)             # [B,64] bool
        occ_e = (enemy.sum(dim=1) > 0.5)                # [B,64] bool
        occupied = (occ_f | occ_e)                      # [B,64] bool

        # piece type index per occupied square, 0..5 in [P,R,B,N,Q,K]
        combined = friendly + enemy                     # [B,6,64], exactly one 1 per occupied square
        type_idx = combined.argmax(dim=1)               # [B,64] long

        # color index per occupied square: 0=friendly, 1=enemy
        color_idx = (occ_e & occupied).long()           # [B,64]; 1 where enemy, else 0

        # embeddings
        type_emb = self.piece_type_embedding(type_idx)  # [B,64,D]
        color_emb = self.color_embedding(color_idx)     # [B,64,D]
        token_pc = type_emb + color_emb                 # [B,64,D]

        # empty squares -> dedicated learnable embedding
        empty_tok = self.empty_embedding.expand(B, 64, -1)  # [B,64,D]
        occ_mask = occupied.unsqueeze(-1)                   # [B,64,1]
        board_tokens = torch.where(occ_mask, token_pc, empty_tok)  # [B,64,D]

        # Global features (e.g., castling planes 12..15 are 1 everywhere if right present)
        if self.global_feature_proj is not None and C > 12:
            # read a single cell; the plane is uniform
            g = x[:, 12:, 0, 0]                 # [B, C-12]
            g_proj = self.global_feature_proj(g)  # [B,D]
        else:
            g_proj = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)

        # Signed material in centipawns: (friendly - enemy) dot piece_values
        # counts per type:
        cnt_f = friendly.sum(dim=2)   # [B,6]
        cnt_e = enemy.sum(dim=2)      # [B,6]
        cp_diff = (cnt_f - cnt_e).to(self.piece_values_cp.dtype) @ self.piece_values_cp  # [B]
        cp_diff = cp_diff.unsqueeze(-1)  # [B,1]
        material_vec = self.material_proj(cp_diff.to(x.dtype))       # [B,D]
        material_proxy_scalar = torch.tanh(cp_diff / self.material_scale_cp)  # [B,1] ~ [-1,1]

        # CLS = learnable + material + global
        cls = self.cls_token.expand(B, -1, -1) + (material_vec + g_proj).unsqueeze(1)  # [B,1,D]
        return board_tokens, cls, material_proxy_scalar

    # ---------- Initialization ----------
    def _init_parameters(self):
        std = 0.02

        def trunc_(t, s):
            if t is None:
                return
            if hasattr(nn.init, "trunc_normal_"):
                # avoid in-place on fp16/bf16 params
                if t.dtype in (torch.float16, torch.bfloat16):
                    t.data = t.data.to(torch.float32)
                nn.init.trunc_normal_(t, std=s, a=-2 * s, b=2 * s)
            else:
                nn.init.normal_(t, std=s)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Embedding)):
                trunc_(getattr(m, "weight", None), std)
                b = getattr(m, "bias", None)
                if b is not None:
                    nn.init.zeros_(b)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, RelativePositionBias):
                trunc_(m.relative_bias_table, std)

        if hasattr(self, "positional_encoding") and hasattr(self.positional_encoding, "absolute_pos_embedding"):
            trunc_(self.positional_encoding.absolute_pos_embedding, std)
        if hasattr(self, "cls_token") and self.cls_token is not None:
            trunc_(self.cls_token, std)
        if hasattr(self, "empty_embedding"):
            trunc_(self.empty_embedding, std)

    def _init_head_specific_weights(self):
        # slightly higher head init std helps heads not be over-attenuated
        std_head = 0.01

        def trunc_(t, s):
            if t is None:
                return
            if hasattr(nn.init, "trunc_normal_"):
                if t.dtype in (torch.float16, torch.bfloat16):
                    t.data = t.data.to(torch.float32)
                nn.init.trunc_normal_(t, std=s, a=-2 * s, b=2 * s)
            else:
                nn.init.normal_(t, std=s)

        if hasattr(self.policy_head, "proj"):
            trunc_(self.policy_head.proj.weight, std_head)
        # value head outputs
        trunc_(self.value_head.scalar_out.weight, std_head)
        trunc_((self.value_head.wdl_out.weight), std_head)

    # ---------- Forward ----------
    def forward(
        self, x,
        value_target=None, policy_target=None, policy_mask=None,
        valueTarget=None, policyTarget=None, policyMask=None
    ):
        # aliasing for backwards compatibility
        if valueTarget is not None: value_target = valueTarget
        if policyTarget is not None: policy_target = policyTarget
        if policyMask  is not None: policy_mask  = policyMask

        B = x.shape[0]

        # Input path
        if self.use_sparse:
            board_tokens, cls, material_proxy = self._process_sparse_input(x)  # [B,64,D], [B,1,D], [B,1]
        else:
            feat = self.input_projection(x)                        # [B,D,8,8]
            board_tokens = feat.flatten(2).transpose(1, 2)         # [B,64,D]
            cls = self.cls_token.expand(B, -1, -1)
            material_proxy = None

        # Add positional encodings to board tokens
        board_tokens = board_tokens + self.positional_encoding(board_tokens)  # [B,64,D]

        # Sequence with CLS first
        seq = torch.cat([cls, board_tokens], dim=1)  # [B,65,D]

        for blk in self.blocks:
            seq = blk(seq)

        seq = self.out_norm(seq)
        cls_feat = seq[:, :1, :]       # [B,1,D]
        board_feat = seq[:, 1:, :]     # [B,64,D]

        # Heads
        value_scalar, wdl_logits = self.value_head(cls_feat)        # scalar in [-1,1], logits [B,3]
        policy_logits = self.policy_head(board_feat)                 # [B,4608]

        # ----------------- Training path -----------------
        if self.training:
            assert value_target is not None and policy_target is not None

            # ----- Value losses -----
            # 1) WDL (piecewise linear mapping from scalar target)
            if self.use_wdl:
                vt = value_target.view(-1)  # [B]
                W = torch.clamp(vt, min=0.0)              # relu(v)
                L = torch.clamp(-vt, min=0.0)             # relu(-v)
                D = torch.clamp(1.0 - W - L, min=0.0)
                wdl_targets = torch.stack([W, D, L], dim=1)  # [B,3]
                wdl_targets = wdl_targets / (wdl_targets.sum(dim=1, keepdim=True) + 1e-8)

                logp = F.log_softmax(wdl_logits, dim=1)
                wdl_loss = -(wdl_targets * logp).sum(dim=1).mean()
            else:
                wdl_loss = 0.0 * value_scalar.mean()  # no-op

            # 2) Scalar MSE to target in [-1,1]
            scalar_loss = self.mse_loss(value_scalar, value_target.view(B, 1))

            # 3) Consistency: (W-L) ≈ scalar
            wdl_probs = F.softmax(wdl_logits, dim=1)
            w_minus_l = (wdl_probs[:, 0:1] - wdl_probs[:, 2:3])  # [B,1]
            calib_loss = self.mse_loss(w_minus_l, value_scalar.detach())  # keep consistency mild

            # 4) Material proxy anchor (tiny)
            if material_proxy is not None:
                material_loss = self.mse_loss(value_scalar, material_proxy)
            else:
                material_loss = 0.0 * value_scalar.mean()

            value_loss = (
                self.wdl_weight * wdl_loss
                + self.scalar_weight * scalar_loss
                + self.calibration_weight * calib_loss
                + self.material_weight * material_loss
            )

            # ----- Policy loss -----
            if policy_target.dim() == 1 or (policy_target.dim() == 2 and policy_target.size(1) == 1):
                # target is a move index (already legal)
                policy_loss = self.cross_entropy_loss(policy_logits, policy_target.view(B))
            else:
                # soft distribution over 4608; mask illegal logits and renormalize targets over legal moves
                logits = policy_logits
                pm = None
                if self.mask_illegal_in_training and policy_mask is not None:
                    pm = policy_mask.view(B, -1).to(dtype=torch.bool)
                    mask_val = -1e4 if logits.dtype == torch.float16 else -1e9
                    logits = logits.masked_fill(~pm, mask_val)

                log_policy = F.log_softmax(logits, dim=1)

                tgt = policy_target
                if pm is not None:
                    # remove any target mass on illegal moves, then renormalize
                    tgt = tgt * pm.to(tgt.dtype)
                    denom = tgt.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    tgt = tgt / denom
                else:
                    # if no mask is provided, just renormalize
                    tgt = tgt / tgt.sum(dim=1, keepdim=True).clamp_min(1e-12)

                policy_loss = -(tgt * log_policy).sum(dim=1).mean()

        # ----------------- Inference path -----------------
        else:
            # Apply legal mask and softmax for policy
            if policy_mask is not None:
                pm = policy_mask.view(B, -1).to(dtype=torch.bool)
                mask_val = -1e4 if policy_logits.dtype == torch.float16 else -1e9
                policy_logits = policy_logits.masked_fill(~pm, mask_val)
            policy_softmax = F.softmax(policy_logits, dim=1)

            # Return scalar value in [-1,1]
            return value_scalar, policy_softmax


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
