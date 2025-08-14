import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. Positional Encoding Modules (ChessPositionalEncoding Restored)
# ==============================================================================

class RelativePositionBias(nn.Module):
    # (Implementation remains the same)
    def __init__(self, num_heads, max_relative_position=7):
        super().__init__()
        self.num_heads = num_heads
        self.max_relative_position = max_relative_position
        num_buckets = 2 * max_relative_position + 1
        self.relative_bias_table = nn.Parameter(
            torch.zeros(num_buckets, num_buckets, num_heads)
        )

    def forward(self, seq_len):
        assert seq_len in (64, 65)
        board_len = 64
        positions = torch.arange(board_len, device=self.relative_bias_table.device)
        ranks = positions // 8
        files = positions % 8
        relative_ranks = ranks.unsqueeze(1) - ranks.unsqueeze(0)
        relative_files = files.unsqueeze(1) - files.unsqueeze(0)
        
        rank_indices = relative_ranks.clamp(-self.max_relative_position, self.max_relative_position) + self.max_relative_position
        file_indices = relative_files.clamp(-self.max_relative_position, self.max_relative_position) + self.max_relative_position
        
        bias = self.relative_bias_table[rank_indices, file_indices]
        bias_64 = bias.permute(2, 0, 1).unsqueeze(0)

        if seq_len == 64:
            return bias_64

        device = bias_64.device
        out = torch.zeros(1, self.num_heads, 65, 65, device=device, dtype=bias_64.dtype)
        out[:, :, 1:, 1:] = bias_64
        return out


class ChessPositionalEncoding(nn.Module):
    """
    Restored Rich Positional Encoding (Rank, File, Diagonal, Absolute).
    Provides explicit spatial information to bootstrap learning.
    """
    def __init__(self, d_model, max_seq_len=64):
        super().__init__()
        self.absolute_pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        self.file_embedding = nn.Embedding(8, d_model)
        self.rank_embedding = nn.Embedding(8, d_model)
        self.diag_embedding = nn.Embedding(15, d_model)
        self.anti_diag_embedding = nn.Embedding(15, d_model)

    def forward(self, x):
        # x is passed to determine device and sequence length
        seq_len = x.shape[1]
        assert seq_len == 64, "This positional encoding is designed for an 8x8 board."

        positions = torch.arange(seq_len, device=x.device)
        files = positions % 8
        ranks = positions // 8
        diagonals = ranks + files
        anti_diagonals = ranks - files + 7

        file_emb = self.file_embedding(files)
        rank_emb = self.rank_embedding(ranks)
        diag_emb = self.diag_embedding(diagonals)
        anti_diag_emb = self.anti_diag_embedding(anti_diagonals)

        total_pos_embedding = (
            self.absolute_pos_embedding[:, :seq_len, :] + 
            file_emb + rank_emb + diag_emb + anti_diag_emb
        )
        
        # Return the encoding itself; addition happens in the main forward pass.
        return total_pos_embedding


# ==============================================================================
# 2. Transformer Core Components
# (Implementations remain the same)
# ==============================================================================

class MultiHeadSelfAttention(nn.Module):
    # (Implementation remains the same)
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
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if seq_len in (64, 65):
            scores = scores + self.relative_position_bias(seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        return output

class GEGLU(nn.Module):
    # (Implementation remains the same)
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.proj = nn.Linear(d_model, d_ff * 2)
        self.out = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return self.out(F.gelu(x1) * x2)

class TransformerBlock(nn.Module):
    # (Implementation remains the same - Pre-LN architecture)
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
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
# 3. Output Heads
# (Implementations remain the same)
# ==============================================================================

class PolicyHead(nn.Module):
    # (Implementation remains the same)
    def __init__(self, d_model, num_move_actions_per_square=72):
        super().__init__()
        self.num_move_actions_per_square = num_move_actions_per_square
        self.proj = nn.Linear(d_model, num_move_actions_per_square)
    def forward(self, x):
        policy_logits = self.proj(x)
        return policy_logits.view(x.shape[0], -1)

class ValueHead(nn.Module):
    # (Implementation remains the same)
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
    # (Implementation remains the same)
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
# 4. Main Model: Titan-Mini (Corrected Architecture)
# ==============================================================================

class TitanMini(nn.Module):
    def __init__(
        self,
        num_layers=13, d_model=512, num_heads=8, d_ff=1920, dropout=0.1,
        policy_weight=1.0, input_planes=112, use_wdl=True, legacy_value_head=False,
    ):
        super().__init__()

        self.policy_weight = policy_weight
        self.d_model = d_model
        self.use_wdl = use_wdl
        self.input_planes = input_planes

        # Adaptive Input Handling Strategy
        self.use_piece_embeddings = (input_planes <= 16)
        
        # Initialize placeholders
        self.input_projection = None; self.piece_type_embedding = None
        self.color_embedding = None; self.cls_token = None
        self.global_feature_proj = None
        # FIX 1: Removed self.input_norm

        if self.use_piece_embeddings:
            # --- Strategy for Sparse Inputs ---
            if input_planes < 12:
                 raise ValueError(f"Sparse input mode requires at least 12 planes, got {input_planes}")

            # 1. Symmetric Embeddings
            self.piece_type_embedding = nn.Embedding(7, d_model) # (Empty=0 + 6 types)
            self.color_embedding = nn.Embedding(2, d_model)      # (Friendly=0, Enemy=1)
            
            # 2. Global Context Projection
            num_global_features = input_planes - 12
            if num_global_features > 0:
                self.global_feature_proj = nn.Linear(num_global_features, d_model)

        else:
            # Strategy for Dense Inputs
            self.input_projection = nn.Conv2d(input_planes, d_model, kernel_size=1)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Shared components
        # FIX 3: Uses the restored Rich ChessPositionalEncoding
        self.positional_encoding = ChessPositionalEncoding(d_model)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(d_model)

        # Output heads and Loss functions
        if use_wdl:
            self.value_head = WDLValueHead(d_model)
        else:
            self.value_head = ValueHead(d_model, legacy_mode=legacy_value_head)
        self.policy_head = PolicyHead(d_model, num_move_actions_per_square=72)
        
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # Initialize parameters (Standardized Truncated Normal)
        self._init_parameters()
        self._init_head_specific_weights()

    def _process_sparse_input(self, x):
        """
        Converts sparse planes into symmetric embeddings.
        Returns combined tokens, type embeddings, and color masks for Signed Material Injection.
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W)
        
        # Based on encoder.py: Friendly (Even Planes), Enemy (Odd Planes)
        friendly_planes = x_flat[:, 0:12:2]
        enemy_planes = x_flat[:, 1:12:2]
        
        # 1. Identify Colors
        is_friendly = torch.sum(friendly_planes, dim=1) > 0.5
        is_enemy = torch.sum(enemy_planes, dim=1) > 0.5
        color_indices = is_enemy.long() # 0=Friendly/Empty, 1=Enemy

        # 2. Identify Piece Types
        combined_planes = friendly_planes + enemy_planes
        type_indices_0_5 = torch.argmax(combined_planes, dim=1)
        is_occupied = (is_friendly + is_enemy) > 0.5
        # Empty=0, P=1..K=6 (based on encoder order)
        piece_type_indices = (type_indices_0_5 + 1) * is_occupied.long()

        # 3. Generate Embeddings
        type_embs = self.piece_type_embedding(piece_type_indices)
        color_embs = self.color_embedding(color_indices)
        
        # 4. Combine
        board_tokens = type_embs + color_embs
        
        # Return components needed for Signed Material Injection
        return board_tokens, type_embs, is_friendly.float(), is_enemy.float()

        
    def _init_parameters(self):
        # (Standardized Truncated Normal initialization remains the same)
        std_dev = 0.02
        def trunc_normal_(tensor, std):
            if tensor is None: return
            if hasattr(nn.init, 'trunc_normal_'):
                # Ensure tensor is float before in-place operation if using AMP
                if tensor.dtype in (torch.float16, torch.bfloat16):
                     tensor.data = tensor.data.to(torch.float32)
                nn.init.trunc_normal_(tensor, std=std, a=-2*std, b=2*std)
            else:
                nn.init.normal_(tensor, std=std)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Embedding)):
                trunc_normal_(getattr(m, 'weight', None), std_dev)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None: nn.init.constant_(m.weight, 1.0)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, RelativePositionBias):
                trunc_normal_(m.relative_bias_table, std_dev)

        if hasattr(self.positional_encoding, 'absolute_pos_embedding'):
             trunc_normal_(self.positional_encoding.absolute_pos_embedding, std_dev)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std_dev)
        

    def _init_head_specific_weights(self):
        # FIX 2: Increased std_dev_head from 0.002 to 0.01 to prevent attenuation.
        std_dev_head = 0.01 
        
        def trunc_normal_(tensor, std):
            if tensor is None: return
            if hasattr(nn.init, 'trunc_normal_'):
                # Ensure tensor is float before in-place operation if using AMP
                if tensor.dtype in (torch.float16, torch.bfloat16):
                     tensor.data = tensor.data.to(torch.float32)
                nn.init.trunc_normal_(tensor, std=std, a=-2*std, b=2*std)
            else:
                nn.init.normal_(tensor, std=std)

        if hasattr(self.policy_head, 'proj') and isinstance(self.policy_head.proj, nn.Linear):
            trunc_normal_(self.policy_head.proj.weight, std_dev_head)

        if self.use_wdl:
            if hasattr(self.value_head, 'value_proj') and isinstance(self.value_head.value_proj, nn.Sequential):
                final_layer = self.value_head.value_proj[-1]
                if isinstance(final_layer, nn.Linear):
                    trunc_normal_(final_layer.weight, std_dev_head)
        else:
            final_layer = None
            if hasattr(self.value_head, 'legacy_mode'):
                if self.value_head.legacy_mode and hasattr(self.value_head, 'fc2'):
                    final_layer = self.value_head.fc2
                elif not self.value_head.legacy_mode and hasattr(self.value_head, 'fc3'):
                    final_layer = self.value_head.fc3
            if final_layer is not None:
                trunc_normal_(final_layer.weight, std_dev_head)

    
    def forward(self, x, value_target=None, policy_target=None, policy_mask=None, 
                valueTarget=None, policyTarget=None, policyMask=None):
        # Handle compatibility
        if valueTarget is not None: value_target = valueTarget
        if policyTarget is not None: policy_target = policyTarget
        if policyMask is not None: policy_mask = policyMask
            
        B = x.shape[0]
        
        # 1. Input Processing (Adaptive Strategy)
        if self.use_piece_embeddings:
            # --- Sparse Input Path (Integrated Approach) ---
            
            # 1a. Generate Symmetric Embeddings and Masks
            tokens_symmetric, type_embs, is_friendly, is_enemy = self._process_sparse_input(x)

            # 1b. Handle Global Features
            global_proj = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
            if self.global_feature_proj is not None:
                global_features = x[:, 12:, 0, 0]
                global_proj = self.global_feature_proj(global_features) # [B, d_model]

            # 1c. Signed Material Injection into CLS
            
            # Calculate Sign Mask: +1 for Friendly, -1 for Enemy
            # Ensure calculation is robust for AMP (Mixed Precision)
            calc_dtype = torch.float32 if x.dtype == torch.float16 else x.dtype
            
            sign_mask = (is_friendly.to(calc_dtype) - is_enemy.to(calc_dtype)).unsqueeze(-1) # [B, 64, 1]

            # Calculate Signed Material Sum (using only Type embeddings)
            signed_material_sum = torch.sum(type_embs.to(calc_dtype) * sign_mask, dim=1) # [B, d_model]
            
            # Cast back to required dtype
            signed_material_sum = signed_material_sum.to(x.dtype)

            # Combine CLS components (Signed Material Sum + Global Context)
            cls = (signed_material_sum + global_proj).unsqueeze(1) # [B, 1, d_model]

            # 1d. Enhance Board Tokens with Global Context
            board_tokens = tokens_symmetric + global_proj.unsqueeze(1)

        else:
            # --- Dense Input Path (Remains the same) ---
            board_tokens = self.input_projection(x)
            board_tokens = board_tokens.flatten(2).transpose(1, 2)
            cls = self.cls_token.expand(B, -1, -1)

        
        # 2. Add positional encodings (Rich version used).
        pos_encoding = self.positional_encoding(board_tokens)
        board_tokens = board_tokens + pos_encoding

        # FIX 1: Removed Input Stabilization (LayerNorm) here. 
        # Stabilization is handled by Pre-LN in the Transformer Blocks.
        
        # 3. Combine CLS token and board tokens.
        x_combined = torch.cat([cls, board_tokens], dim=1)
        
        # 4. Transformer blocks.
        for block in self.transformer_blocks:
            x_combined = block(x_combined)
        
        # 5. Final normalization.
        x_combined = self.output_norm(x_combined)

        # 6. Split features and compute outputs.
        cls_features = x_combined[:, 0:1, :]
        board_features = x_combined[:, 1:, :]

        value = self.value_head(cls_features)
        policy = self.policy_head(board_features)
        
        # --- Training Path (Loss calculations retained from previous fixes) ---
        if self.training:
            assert value_target is not None and policy_target is not None
            
            # Value Loss Calculation (Piecewise Linear WDL)
            if self.use_wdl:
                if value_target.dim() == 2 and value_target.shape[1] == 3:
                    wdl_targets = value_target
                else:
                    value_flat = value_target.view(-1)
                    W = torch.relu(value_flat); L = torch.relu(-value_flat); D = 1.0 - W - L
                    D = torch.clamp(D, min=0.0)
                    wdl_targets = torch.stack([W, D, L], dim=1)

                log_probs = F.log_softmax(value, dim=1)
                wdl_targets = wdl_targets / (wdl_targets.sum(dim=1, keepdim=True) + 1e-8)
                value_loss = -(wdl_targets * log_probs).sum(dim=1).mean()
            else:
                value_loss = self.mse_loss(value, value_target)
            
            # Policy Loss Calculation (Cross-Entropy for soft targets)
            if policy_target.dim() == 1 or (policy_target.dim() == 2 and policy_target.size(1) == 1):
                policy_target_1d = policy_target.view(B)
                policy_loss = self.cross_entropy_loss(policy, policy_target_1d)
            else:
                policy_target_flat = policy_target.view(B, -1)
                log_policy = F.log_softmax(policy, dim=1)
                policy_target_flat = policy_target_flat + 1e-8
                policy_target_flat = policy_target_flat / policy_target_flat.sum(dim=1, keepdim=True)
                policy_loss = -(policy_target_flat * log_policy).sum(dim=1).mean()
            
            total_loss = value_loss + self.policy_weight * policy_loss
            return total_loss, value_loss, policy_loss
        
        # --- Inference Path ---
        else:
            if policy_mask is not None:
                policy_mask = policy_mask.view(B, -1)
                policy = policy.masked_fill(policy_mask == 0, -1e9)
            
            policy_softmax = F.softmax(policy, dim=1)
            
            if self.use_wdl:
                wdl_probs = F.softmax(value, dim=1)
                value_scalar = wdl_probs[:, 0:1] - wdl_probs[:, 2:3]
                return value_scalar, policy_softmax
            else:
                return value, policy_softmax


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)