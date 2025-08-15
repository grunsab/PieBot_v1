# TitanMiniNetwork.py (Revised with Fixes and Auxiliary Losses)
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
    # (Implementation remains as provided by user)
    def __init__(self, num_heads, max_relative_position=7, bias_scale=2.0):
        super().__init__()
        self.num_heads = num_heads
        self.max_relative_position = max_relative_position
        self.bias_scale = float(bias_scale)
        num_buckets = 2 * max_relative_position + 1
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
        
        bias = torch.tanh(self.relative_bias_table[r_idx, f_idx]) * self.bias_scale
        bias_64 = bias.permute(2, 0, 1).unsqueeze(0)

        if seq_len == 64:
            return bias_64

        out = torch.zeros(1, self.num_heads, 65, 65, device=device, dtype=bias_64.dtype)
        out[:, :, 1:, 1:] = bias_64
        return out


class ChessPositionalEncoding(nn.Module):
    """
    Rich positional encoding: absolute + file/rank/diag/anti-diag.
    """
    # (Implementation remains as provided by user)
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
    # (Implementation remains as provided by user, updated autocast call for compatibility)
    def __init__(self, d_model, num_heads, dropout=0.1, attn_clamp: float = 50.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model); self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model); self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relative_position_bias = RelativePositionBias(num_heads)
        self.attn_clamp = float(attn_clamp)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        orig_dtype = x.dtype
        device_type = x.device.type

        # Disable autocast for attention math; do it in fp32
        # Use torch.autocast(device_type=...) for broader compatibility
        with torch.autocast(device_type=device_type, enabled=False):
            xf = x.float()
            Q = self.W_q(xf).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(xf).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(xf).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            if T in (64, 65):
                scores = scores + self.relative_position_bias(T)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e4)

            if self.attn_clamp is not None and self.attn_clamp > 0:
                scores = scores.clamp(min=-self.attn_clamp, max=self.attn_clamp)

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            context = torch.matmul(attn, V)
            context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)

            out = self.W_o(context).to(orig_dtype)

        return out

class GEGLU(nn.Module):
    # (Implementation remains as provided by user)
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
    # (Implementation remains as provided by user)
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = GEGLU(d_model, d_ff)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Stabilization occurs here via Pre-LN
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        return x

# ==============================================================================
# 3) Output Heads (Modified for Auxiliary Losses)
# ==============================================================================

class PolicyHead(nn.Module):
    # (Implementation remains as provided by user)
    def __init__(self, d_model, num_move_actions_per_square=72, logit_scale=1.0):
        super().__init__()
        self.num_move_actions_per_square = num_move_actions_per_square
        self.proj = nn.Linear(d_model, num_move_actions_per_square)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale)))

    def forward(self, x):
        # x: [B, 64, D]
        logits_sq_plane = self.proj(x) * self.logit_scale      # [B, 64, 72]
        logits_plane_sq = logits_sq_plane.permute(0, 2, 1).contiguous()  # [B, 72, 64]
        logits = logits_plane_sq.view(x.shape[0], -1)          # [B, 4608] PLANE-MAJOR
        return logits

# NEW: Auxiliary Calibration Head
class CalibrationHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)
        self.output_activation = nn.Tanh()
    
    def forward(self, x):
        return self.output_activation(self.proj(x))

class ValueHead(nn.Module):
    # Modified to support Calibration Loss
    def __init__(self, d_model, legacy_mode=False, use_calibration=True):
        super().__init__()
        self.legacy_mode = legacy_mode
        self.use_calibration = use_calibration
        self.calibration_head = None

        if legacy_mode:
            self.fc1 = nn.Linear(d_model, d_model // 2); self.activation1 = nn.GELU(); self.dropout1 = nn.Dropout(0.1)
            self.fc2 = nn.Linear(d_model // 2, 1); self.output_activation = nn.Tanh()
            if self.use_calibration:
                self.calibration_head = CalibrationHead(d_model // 2)
        else:
            self.fc1 = nn.Linear(d_model, d_model // 2); self.activation1 = nn.GELU(); self.dropout1 = nn.Dropout(0.1)
            self.fc2 = nn.Linear(d_model // 2, 128); self.activation2 = nn.GELU(); self.dropout2 = nn.Dropout(0.1)
            self.fc3 = nn.Linear(128, 1); self.output_activation = nn.Tanh()
            if self.use_calibration:
                self.calibration_head = CalibrationHead(128)

    def forward(self, x):
        x = x.squeeze(1)
        intermediate_features = None

        if self.legacy_mode:
            x = self.fc1(x); x = self.activation1(x); x = self.dropout1(x)
            if self.use_calibration: intermediate_features = x
            x = self.fc2(x); x = self.output_activation(x)
        else:
            x = self.fc1(x); x = self.activation1(x); x = self.dropout1(x)
            x = self.fc2(x); x = self.activation2(x); x = self.dropout2(x)
            if self.use_calibration: intermediate_features = x
            x = self.fc3(x); x = self.output_activation(x)
        
        if self.training and self.use_calibration and intermediate_features is not None:
            calibration_output = self.calibration_head(intermediate_features)
            return x, calibration_output
        
        return x

class WDLValueHead(nn.Module):
    # Modified to support Calibration Loss (Rewritten from Sequential)
    def __init__(self, d_model, use_calibration=True):
        super().__init__()
        self.use_calibration = use_calibration
        self.calibration_head = None

        # Define layers explicitly
        self.fc1 = nn.Linear(d_model, d_model // 2); self.act1 = nn.GELU(); self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(d_model // 2, 128); self.act2 = nn.GELU(); self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, 3)

        if self.use_calibration:
             self.calibration_head = CalibrationHead(128)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.fc1(x); x = self.act1(x); x = self.drop1(x)
        x = self.fc2(x); x = self.act2(x); x = self.drop2(x)
        
        intermediate_features = x
        x = self.fc3(x)

        if self.training and self.use_calibration:
            calibration_output = self.calibration_head(intermediate_features)
            return x, calibration_output

        return x

# ==============================================================================
# 4) Titan-Mini (Revised with Fixes and Auxiliary Losses)
# ==============================================================================

# Standard Centipawn Values (Order: Empty, P, R, B, N, Q, K)
# Matches the order derived from encoder.py (P, R, B, N, Q, K)
# CP_VALUES = torch.tensor([0., 100., 500., 320., 300., 900., 20000.], dtype=torch.float32)
PAWN_UNITS = torch.tensor([0., 1., 5., 3., 3., 9., 0.], dtype=torch.float32)

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

        self.input_norm = nn.LayerNorm(d_model, eps=1e-5)


        # NEW: Configuration for Auxiliary Losses (Tuned by train_titan_mini.py)
        # These are default weights; they will be dynamically adjusted during training.
        self.material_weight = 0.05
        self.calibration_weight = 0.25
        self.wdl_weight = 0.60 # Weight for the main WDL/Value loss

        # Sparse input path (16 planes)
        if input_planes < 12:
            raise ValueError(f"Input planes must be >=12, got {input_planes}")
        self.piece_type_embedding = nn.Embedding(7, d_model)
        self.color_embedding = nn.Embedding(2, d_model)

        # NEW: Material Projection (for Anchor Loss)
        self.material_proj = nn.Linear(d_model, 1)
        # # Register CP_VALUES as a buffer so it moves with the model
        # self.register_buffer('cp_values_target', CP_VALUES)
        self.register_buffer('piece_value_anchor', PAWN_UNITS / 9.0)  # -> in [0,1]


        num_global_features = max(0, input_planes - 12)
        self.global_feature_proj = nn.Linear(num_global_features or 1, d_model) if num_global_features > 0 else None

        self.positional_encoding = ChessPositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # FIX 1: Removed self.input_norm. Stabilization relies on Pre-LN in TransformerBlock.
        # self.input_norm = nn.LayerNorm(d_model, eps=1e-5) 
        
        self.output_norm = nn.LayerNorm(d_model, eps=1e-5)

        # Value Head (initialized with support for Calibration Loss)
        use_calibration = True
        if use_wdl:
            self.value_head = WDLValueHead(d_model, use_calibration=use_calibration)
        else:
            self.value_head = ValueHead(d_model, legacy_mode=legacy_value_head, use_calibration=use_calibration)
        
        self.policy_head = PolicyHead(d_model, num_move_actions_per_square=72, logit_scale=1.0)

        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self._init_parameters()
        self._init_head_specific_weights()

    # ---- sparse input helpers ----
    def _process_sparse_input(self, x):
        # (Implementation remains the same)
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

        type_embs = self.piece_type_embedding(piece_type_indices)
        color_embs = self.color_embedding(color_indices)
        board_tokens = type_embs + color_embs

        return board_tokens, type_embs, is_friendly.float(), is_enemy.float()

    # ---- init ----
    def _init_parameters(self):
        # (Implementation remains the same, ensuring robust initialization)
        std = 0.02
        def trunc_normal_(t, s):
            if t is None: return
            if hasattr(nn.init, "trunc_normal_"):
                # Ensure tensor is float32 before in-place init if using AMP
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
        # (Updated to initialize auxiliary heads)
        std_head = 0.01
        def trunc_normal_(t, s):
            if t is None: return
            if hasattr(nn.init, "trunc_normal_"):
                # Ensure tensor is float32 before in-place init if using AMP
                if t.dtype in (torch.float16, torch.bfloat16):
                    t.data = t.data.to(torch.float32)
                nn.init.trunc_normal_(t, std=s, a=-2*s, b=2*s)
            else:
                nn.init.normal_(t, std=s)

        if hasattr(self.policy_head, 'proj'):
            trunc_normal_(self.policy_head.proj.weight, std_head)
        
        # Initialize auxiliary heads
        if hasattr(self, 'material_proj'):
            trunc_normal_(self.material_proj.weight, std_head)

        # Final value layers and calibration heads
        if self.use_wdl:
            # WDLValueHead now uses explicit fc3
            if hasattr(self.value_head, 'fc3'):
                 trunc_normal_(self.value_head.fc3.weight, std_head)
            if hasattr(self.value_head, 'calibration_head') and self.value_head.calibration_head:
                 trunc_normal_(self.value_head.calibration_head.proj.weight, std_head)
        else:
            if hasattr(self.value_head, 'legacy_mode'):
                final_layer = self.value_head.fc2 if self.value_head.legacy_mode else self.value_head.fc3
                if final_layer:
                    trunc_normal_(final_layer.weight, std_head)
            if hasattr(self.value_head, 'calibration_head') and self.value_head.calibration_head:
                 trunc_normal_(self.value_head.calibration_head.proj.weight, std_head)

    # ---- forward ----
    def forward(self, x, value_target=None, policy_target=None, policy_mask=None,
                valueTarget=None, policyTarget=None, policyMask=None, return_logits: bool = False):
        # (Backward-compat kwargs remain the same)
        if valueTarget is not None: value_target = valueTarget
        if policyTarget is not None: policy_target = policyTarget
        if policyMask is not None: policy_mask = policyMask

        B = x.shape[0]
        # Sparse input path
        tokens_symmetric, type_embs, is_friendly, is_enemy = self._process_sparse_input(x)

        # Global context
        global_proj = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
        if self.global_feature_proj is not None:
            global_features = x[:, 12:, 0, 0]
            global_proj = self.global_feature_proj(global_features)

        # Signed material sum for CLS (compute in fp32 for stability)
        calc_dtype = torch.float32 if x.dtype == torch.float16 else x.dtype
        sign_mask = (is_friendly.to(calc_dtype) - is_enemy.to(calc_dtype)).unsqueeze(-1)
        signed_material_sum = torch.sum(type_embs.to(calc_dtype) * sign_mask, dim=1)
        cls = (signed_material_sum.to(x.dtype) + global_proj).unsqueeze(1)

        # Enhance board tokens with global context
        board_tokens = tokens_symmetric + global_proj.unsqueeze(1)

        # Positional encodings
        pos = self.positional_encoding(board_tokens)
        board_tokens = board_tokens + pos

        # Combine
        x_combined = torch.cat([cls, board_tokens], dim=1)
        
        # FIX 1: Removed input_norm application.
        x_combined = self.input_norm(x_combined)

        # Transformer stack
        for block in self.transformer_blocks:
            x_combined = block(x_combined, mask=None)

        x_combined = self.output_norm(x_combined)
        cls_features = x_combined[:, 0:1, :]
        board_features = x_combined[:, 1:, :]

        # Heads
        # Value head might return (value, calibration_output) during training
        value_output = self.value_head(cls_features)
        calibration_output = None
        if self.training and isinstance(value_output, tuple):
            value, calibration_output = value_output
        else:
            value = value_output
        
        policy = self.policy_head(board_features)

        # ---- TRAINING PATH ----
        if self.training:
            assert value_target is not None and policy_target is not None

            # (Policy masking and finite checks remain the same)
            if policy_mask is not None:
                pm = policy_mask.view(B, -1)
                policy = policy.masked_fill(pm == 0, -1e4)

            if not torch.isfinite(policy).all():
                # Use slightly smaller constants for posinf/neginf for safety
                policy = torch.nan_to_num(policy, nan=0.0, posinf=5e3, neginf=-5e3)

            # ----- Main Value loss -----
            # (WDL/Value loss calculation remains the same - computed in FP32)
            if self.use_wdl:
                if value_target.dim() == 2 and value_target.size(1) == 3:
                    wdl_targets = value_target
                else:
                    v = value_target.view(-1)
                    W = torch.relu(v); L = torch.relu(-v); D = torch.clamp(1.0 - W - L, min=0.0)
                    wdl_targets = torch.stack([W, D, L], dim=1)
                
                # Calculate loss in FP32
                log_probs = F.log_softmax(value.float(), dim=1)
                wdl_targets = wdl_targets / (wdl_targets.sum(dim=1, keepdim=True) + 1e-8)
                value_loss = -(wdl_targets * log_probs).sum(dim=1).mean()
            else:
                value_loss = self.mse_loss(value.float(), value_target.float())

            # ----- Policy loss -----
            # (Policy loss calculation remains the same - computed in FP32)
            if policy_target.dim() == 1 or (policy_target.dim() == 2 and policy_target.size(1) == 1):
                policy_loss = self.cross_entropy_loss(policy, policy_target.view(B))
            else:
                pt = policy_target.view(B, -1).float()
                pt = pt + 1e-8
                pt = pt / pt.sum(dim=1, keepdim=True)
                logp = F.log_softmax(policy.float(), dim=1)
                policy_loss = -(pt * logp).sum(dim=1).mean()

            # ----- Auxiliary Losses (NEW) -----
            
            # 1. Material Anchor Loss
            # material_loss = torch.tensor(0.0, device=x.device, dtype=value_loss.dtype)
            # if self.material_weight > 0:
            #     # Project embeddings to scalars
            #     projected_embeddings = self.material_proj(self.piece_type_embedding.weight).squeeze(1)
            #     # Calculate MSE loss against target CP values (ensure FP32)
            #     material_loss = F.mse_loss(projected_embeddings.float(), self.cp_values_target)

            material_loss = torch.tensor(0.0, device=x.device, dtype=value_loss.dtype)
            if self.material_weight > 0:
                # [7, D] -> [7, 1] -> [7]; bound to [0,1]
                pred = torch.sigmoid(self.material_proj(self.piece_type_embedding.weight)).squeeze(1).float()
                target = self.piece_value_anchor  # [7] in [0,1]
                # Optionally ignore index 0 (Empty) and 6 (King)
                idx = torch.tensor([1, 2, 3, 4, 5], device=pred.device)
                material_loss = F.mse_loss(pred.index_select(0, idx), target.index_select(0, idx))


            # 2. Calibration Loss
            calibration_loss = torch.tensor(0.0, device=x.device, dtype=value_loss.dtype)
            if self.calibration_weight > 0 and calibration_output is not None:
                 # Determine the scalar target
                if self.use_wdl:
                    if value_target.dim() == 2 and value_target.size(1) == 3:
                        # Convert WDL targets back to scalar: E = W - L
                        scalar_target = value_target[:, 0] - value_target[:, 2]
                    else:
                        scalar_target = value_target.view(-1)
                else:
                    scalar_target = value_target.view(-1)
                
                # Calibration loss uses MSE against the scalar target (ensure FP32)
                calibration_loss = F.mse_loss(calibration_output.view(-1).float(), scalar_target.float())

            # Total Loss Combination
            total_loss = (self.wdl_weight * value_loss + 
                          self.policy_weight * policy_loss + 
                          self.material_weight * material_loss +
                          self.calibration_weight * calibration_loss)


            # Final finite guard (remains the same)
            if not torch.isfinite(total_loss):
                # Ensure the zero tensor requires grad and matches loss dtype (FP32)
                total_loss = torch.zeros((), device=x.device, dtype=value_loss.dtype, requires_grad=True)
                value_loss = total_loss
                policy_loss = total_loss

            # Return the primary losses (value_loss, policy_loss) for monitoring consistency
            return total_loss, value_loss, policy_loss

        # ---- INFERENCE PATH ----
        else:
            # (Implementation remains the same)
            if policy_mask is not None:
                pm = policy_mask.view(B, -1)
                policy = policy.masked_fill(pm == 0, -1e4)
            if return_logits:
                # return raw logits (value logits if WDL, else scalar; and policy logits)
                return (value, policy)
            # default: probabilities / scalar
            policy_softmax = F.softmax(policy, dim=1)
            if self.use_wdl:
                wdl = F.softmax(value, dim=1)
                value_scalar = wdl[:, 0:1] - wdl[:, 2:3]
                return value_scalar, policy_softmax
            else:
                return value, policy_softmax

            # if policy_mask is not None:
            #     pm = policy_mask.view(B, -1)
            #     policy = policy.masked_fill(pm == 0, -1e4)
            # policy_softmax = F.softmax(policy, dim=1)

            # if self.use_wdl:
            #     wdl = F.softmax(value, dim=1)
            #     value_scalar = wdl[:, 0:1] - wdl[:, 2:3]
            #     return value_scalar, policy_softmax
            # else:
            #     return value, policy_softmax


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)