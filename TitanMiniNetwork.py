import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. Positional Encoding Modules
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
        nn.init.normal_(self.relative_bias_table, std=0.02)

    def forward(self, seq_len):
        assert seq_len in (64, 65)
        board_len = 64
        positions = torch.arange(board_len, device=self.relative_bias_table.device)
        ranks = positions // 8
        files = positions % 8
        relative_ranks = ranks.unsqueeze(1) - ranks.unsqueeze(0)
        relative_files = files.unsqueeze(1) - files.unsqueeze(0)
        clipped_rel_ranks = relative_ranks.clamp(-self.max_relative_position, self.max_relative_position)
        clipped_rel_files = relative_files.clamp(-self.max_relative_position, self.max_relative_position)
        rank_indices = clipped_rel_ranks + self.max_relative_position
        file_indices = clipped_rel_files + self.max_relative_position
        bias = self.relative_bias_table[rank_indices, file_indices]
        bias_64 = bias.permute(2, 0, 1).unsqueeze(0)

        if seq_len == 64:
            return bias_64

        device = bias_64.device
        out = torch.zeros(1, self.num_heads, 65, 65, device=device, dtype=bias_64.dtype)
        out[:, :, 1:, 1:] = bias_64
        return out


class ChessPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=64):
        super().__init__()
        self.absolute_pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        self.file_embedding = nn.Embedding(8, d_model)
        self.rank_embedding = nn.Embedding(8, d_model)
        self.diag_embedding = nn.Embedding(15, d_model)
        self.anti_diag_embedding = nn.Embedding(15, d_model)

    def forward(self, x):
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
        return x + total_pos_embedding


# ==============================================================================
# 2. Transformer Core Components
# (MultiHeadSelfAttention, GEGLU, TransformerBlock remain unchanged from the prompt)
# ==============================================================================

class MultiHeadSelfAttention(nn.Module):
    # (Implementation remains the same as provided in the prompt)
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
    # (Implementation remains the same as provided in the prompt)
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.proj = nn.Linear(d_model, d_ff * 2)
        self.out = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return self.out(F.gelu(x1) * x2)

class TransformerBlock(nn.Module):
    # (Implementation remains the same as provided in the prompt)
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = GEGLU(d_model, d_ff)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        return x


# ==============================================================================
# 3. Output Heads (ValueHead modified)
# ==============================================================================

class PolicyHead(nn.Module):
    """
    Predicts the policy (a probability distribution over all possible moves).
    """
    def __init__(self, d_model, num_move_actions_per_square=72):
        super().__init__()
        self.num_move_actions_per_square = num_move_actions_per_square
        self.proj = nn.Linear(d_model, num_move_actions_per_square)

    def forward(self, x):
        # Input x shape: [batch, 64, d_model]
        policy_logits = self.proj(x)  # [batch, 64, num_move_actions_per_square]
        # Reshape to a flat policy vector: [batch, 64 * num_move_actions_per_square]
        return policy_logits.view(x.shape[0], -1)


class ValueHead(nn.Module):
    """
    Predicts the value of the position using the special [CLS] token.
    Uses Tanh activation (like AlphaZero).
    """
    def __init__(self, d_model, legacy_mode=False):
        super().__init__()
        self.legacy_mode = legacy_mode
        self.d_model = d_model
        
        if legacy_mode:
            # Legacy architecture with 2 layers
            self.fc1 = nn.Linear(d_model, d_model // 2)
            self.activation1 = nn.GELU()
            self.dropout1 = nn.Dropout(0.1)
            self.fc2 = nn.Linear(d_model // 2, 1)
            self.output_activation = nn.Tanh()
            
            # FIX: Removed explicit initialization from here. 
            # Initialization is now handled by the main model (TitanMini).

        else:
            # New architecture with 3 layers
            self.fc1 = nn.Linear(d_model, d_model // 2)
            self.activation1 = nn.GELU()
            self.dropout1 = nn.Dropout(0.1)
            self.fc2 = nn.Linear(d_model // 2, 128)
            self.activation2 = nn.GELU()
            self.dropout2 = nn.Dropout(0.1)
            self.fc3 = nn.Linear(128, 1)
            self.output_activation = nn.Tanh()
            
            # FIX: Removed explicit initialization from here.
        
    def forward(self, x):
        # Input x is the [CLS] token feature: [batch, 1, d_model]
        x = x.squeeze(1)
        
        if self.legacy_mode:
            x = self.fc1(x)
            x = self.activation1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.output_activation(x)
        else:
            x = self.fc1(x)
            x = self.activation1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.activation2(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            x = self.output_activation(x)
        
        return x


class WDLValueHead(nn.Module):
    """
    Win-Draw-Loss (WDL) value head. Predicts three probabilities: win, draw, and loss.
    """
    def __init__(self, d_model):
        super().__init__()
        self.value_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),  # Output 3 logits for Win, Draw, Loss
        )
        
    def forward(self, x):
        # Input x is the [CLS] token feature: [batch, 1, d_model]
        return self.value_proj(x.squeeze(1))


# ==============================================================================
# 4. Main Model: Titan-Mini (Modified init, forward, and added initialization helper)
# ==============================================================================

class TitanMini(nn.Module):
    """
    A smaller, CPU-friendly transformer-based neural network for chess.
    """
    def __init__(
        self,
        num_layers=13,
        d_model=512,
        num_heads=8,
        d_ff=1920,
        dropout=0.1,
        policy_weight=1.0,
        input_planes=112,
        use_wdl=True,
        legacy_value_head=False,
    ):
        super().__init__()

        self.policy_weight = policy_weight
        self.d_model = d_model
        self.use_wdl = use_wdl

        self.input_projection = nn.Conv2d(input_planes, d_model, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional_encoding = ChessPositionalEncoding(d_model)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(d_model)

        if use_wdl:
            self.value_head = WDLValueHead(d_model)
        else:
            self.value_head = ValueHead(d_model, legacy_mode=legacy_value_head)
        self.policy_head = PolicyHead(d_model, num_move_actions_per_square=72)
        
        self.mse_loss = nn.MSELoss()
        # self.bce_with_logits_loss = nn.BCEWithLogitsLoss() # Removed as it was unused
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # Initialize parameters
        self._init_parameters()
        # FIX: Apply specific initialization for heads *after* generic initialization.
        self._init_head_specific_weights()
        
    def _init_parameters(self):
        """Initialize parameters with a generic strategy (Xavier Uniform)."""
        for p in self.parameters():
            if p.dim() > 1:
                # Note: This applies Xavier to Embeddings as well. While sometimes suboptimal 
                # (Normal distribution is often preferred for embeddings), we keep it for consistency 
                # with the original intent unless specific issues arise.
                nn.init.xavier_uniform_(p)

        nn.init.normal_(self.cls_token, std=0.02)

    def _init_head_specific_weights(self):
        """
        Applies specific initialization for the policy and value heads.
        This ensures the final layers have small weights and zero bias to prevent saturation (Softmax/Tanh)
        at the beginning of training.
        """
        
        # 1. Policy Head Initialization (Softmax output)
        if hasattr(self.policy_head, 'proj') and isinstance(self.policy_head.proj, nn.Linear):
            # Use a small gain (0.1) for stability.
            nn.init.xavier_uniform_(self.policy_head.proj.weight, gain=0.1)
            if self.policy_head.proj.bias is not None:
                nn.init.zeros_(self.policy_head.proj.bias)

        # 2. Value Head Initialization
        if self.use_wdl:
            # WDLValueHead (Softmax output)
            if hasattr(self.value_head, 'value_proj') and isinstance(self.value_head.value_proj, nn.Sequential):
                # Ensure biases are zero throughout the head
                for m in self.value_head.value_proj.modules():
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.zeros_(m.bias)

                # Initialize final layer with small gain
                final_layer = self.value_head.value_proj[-1]
                if isinstance(final_layer, nn.Linear):
                    nn.init.xavier_uniform_(final_layer.weight, gain=0.1)

        else:
            # Standard ValueHead (Tanh output)
            # We use xavier_normal_ here as it is generally preferred for Tanh activations.
            
            # Initialize intermediate layers (override generic initialization and set bias to zero)
            if hasattr(self.value_head, 'fc1'):
                nn.init.xavier_normal_(self.value_head.fc1.weight)
                if self.value_head.fc1.bias is not None:
                    nn.init.zeros_(self.value_head.fc1.bias)
            
            # Handle fc2 depending on mode
            if hasattr(self.value_head, 'fc2') and hasattr(self.value_head, 'legacy_mode'):
                if not self.value_head.legacy_mode: # Only initialize if it's not the final layer
                    nn.init.xavier_normal_(self.value_head.fc2.weight)
                    if self.value_head.fc2.bias is not None:
                        nn.init.zeros_(self.value_head.fc2.bias)

            # Initialize final layer with small gain
            final_layer = None
            if hasattr(self.value_head, 'legacy_mode'):
                if self.value_head.legacy_mode and hasattr(self.value_head, 'fc2'):
                    final_layer = self.value_head.fc2
                elif not self.value_head.legacy_mode and hasattr(self.value_head, 'fc3'):
                    final_layer = self.value_head.fc3
            
            if final_layer is not None:
                # Override generic initialization with xavier_normal_ and small gain
                nn.init.xavier_normal_(final_layer.weight, gain=0.1)
                if final_layer.bias is not None:
                    nn.init.zeros_(final_layer.bias)

    
    def forward(self, x, value_target=None, policy_target=None, policy_mask=None, 
                valueTarget=None, policyTarget=None, policyMask=None):
        # Handle compatibility
        if valueTarget is not None: value_target = valueTarget
        if policyTarget is not None: policy_target = policyTarget
        if policyMask is not None: policy_mask = policyMask
            
        batch_size = x.shape[0]

        # Validate input shape
        if len(x.shape) != 4 or x.shape[2] != 8 or x.shape[3] != 8:
            raise ValueError(f"Expected input shape [batch, planes, 8, 8], got {x.shape}")
        
        # 1. Project input planes and reshape.
        x = self.input_projection(x)  # [batch, d_model, 8, 8]
        x = x.flatten(2).transpose(1, 2)  # [batch, 64, d_model]
        
        # 2. Add positional encodings.
        x = self.positional_encoding(x)
        
        # 3. Prepend the [CLS] token.
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [batch, 65, d_model]
        
        # 4. Transformer blocks.
        for block in self.transformer_blocks:
            x = block(x)
        
        # 5. Final normalization.
        x = self.output_norm(x)

        # 6. Split features.
        cls_features = x[:, 0:1, :]   # [batch, 1, d_model]
        board_features = x[:, 1:, :] # [batch, 64, d_model]

        # 7. Compute value and policy.
        value = self.value_head(cls_features)
        policy = self.policy_head(board_features)
        
        # --- Training Path ---
        if self.training:
            assert value_target is not None and policy_target is not None
            
            # Value Loss Calculation
            if self.use_wdl:
                if value_target.dim() == 2 and value_target.shape[1] == 3:
                    wdl_targets = value_target
                else:
                    # --- FIX 1: Correct WDL Conversion (Piecewise Linear) ---
                    # Convert scalar values V (in [-1, 1]) to WDL probabilities [W, D, L]
                    # This preserves the expected value E = W - L.
                    
                    value_flat = value_target.view(-1)
                    
                    # P(W) = relu(V)
                    W = torch.relu(value_flat)
                    # P(L) = relu(-V)
                    L = torch.relu(-value_flat)
                    # P(D) = 1 - (W + L)
                    D = 1.0 - W - L
                    
                    # Clamp D for numerical stability or if input slightly outside [-1, 1]
                    D = torch.clamp(D, min=0.0)

                    # Assumes WDLValueHead output order is [Win, Draw, Loss]
                    wdl_targets = torch.stack([W, D, L], dim=1)

                # WDL loss using cross entropy with soft targets
                log_probs = F.log_softmax(value, dim=1)
                # Ensure normalization (important if targets came from external source or due to clamping above)
                wdl_targets = wdl_targets / (wdl_targets.sum(dim=1, keepdim=True) + 1e-8)
                # Cross-entropy: -Sum(Target * log(Prediction))
                value_loss = -(wdl_targets * log_probs).sum(dim=1).mean()
            else:
                # For non-WDL mode (Tanh output), use MSE loss
                value_loss = self.mse_loss(value, value_target)
            
            # Policy Loss Calculation
            if policy_target.dim() == 1 or (policy_target.dim() == 2 and policy_target.size(1) == 1):
                # Hard targets (single index)
                policy_target_1d = policy_target.view(batch_size)
                policy_loss = self.cross_entropy_loss(policy, policy_target_1d)
            else:
                # Soft targets (distribution, e.g., MCTS visits)
                policy_target_flat = policy_target.view(batch_size, -1)
                
                # --- FIX 3: Use Cross-Entropy instead of KL Divergence ---
                log_policy = F.log_softmax(policy, dim=1)
                
                # Ensure targets are normalized and stable (as done in AlphaZeroNetwork.py)
                policy_target_flat = policy_target_flat + 1e-8
                policy_target_flat = policy_target_flat / policy_target_flat.sum(dim=1, keepdim=True)

                # Calculate cross-entropy: -Sum(Target * log(Prediction))
                policy_loss = -(policy_target_flat * log_policy).sum(dim=1).mean()
            
            total_loss = value_loss + self.policy_weight * policy_loss
            return total_loss, value_loss, policy_loss
        
        # --- Inference Path ---
        else:
            if policy_mask is not None:
                policy_mask = policy_mask.view(batch_size, -1)
                policy = policy.masked_fill(policy_mask == 0, -1e9)
            
            policy_softmax = F.softmax(policy, dim=1)
            
            # Convert WDL to scalar value for inference
            if self.use_wdl:
                # E[V] = P(W) - P(L)
                wdl_probs = F.softmax(value, dim=1)
                # Ensure order matches training targets: [Win, Draw, Loss]
                value_scalar = wdl_probs[:, 0:1] - wdl_probs[:, 2:3]
                return value_scalar, policy_softmax
            else:
                # Non-WDL mode: value already has Tanh applied (outputs in [-1, 1])
                return value, policy_softmax


def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # --- Model Initialization Test ---
    print("Testing TitanMini initialization (WDL Mode)...")
    model = TitanMini(use_wdl=True)

    num_params = count_parameters(model)
    model_size_mb = num_params * 4 / (1024 * 1024)

    print(f"Number of trainable parameters: {num_params:,}")
    print(f"Estimated model size: {model_size_mb:.2f} MB")
    
    # --- Test Forward Pass ---
    batch_size = 4
    dummy_input = torch.randn(batch_size, 112, 8, 8)
    # Test specific values including extremes for WDL conversion
    dummy_value_target = torch.tensor([[1.0], [-1.0], [0.0], [0.5]])
    
    num_total_moves = 72 * 64
    # Test soft policy targets
    dummy_policy_target_soft = torch.rand(batch_size, num_total_moves)
    dummy_policy_target_soft /= dummy_policy_target_soft.sum(dim=1, keepdim=True)
    
    dummy_mask = torch.randint(0, 2, (batch_size, num_total_moves))

    # --- Training Mode Test (WDL Conversion and Soft Policy Loss) ---
    print("\n--- Testing Training Mode (WDL and Soft Policy) ---")
    model.train()
    total_loss, value_loss, policy_loss = model(
        dummy_input, 
        value_target=dummy_value_target,
        policy_target=dummy_policy_target_soft,
    )
    print(f"Total loss: {total_loss.item():.4f}, Value loss: {value_loss.item():.4f}, Policy loss: {policy_loss.item():.4f}")
    
    # --- Inference Mode Test ---
    print("\n--- Testing Inference Mode ---")
    model.eval()
    with torch.no_grad():
        value, policy = model(dummy_input, policy_mask=dummy_mask)
        print(f"Value output shape: {value.shape}")
        print(f"Value output example (P(W)-P(L)): {value.squeeze().detach().numpy()}")
        # Check initial values are small (due to initialization fix)
        if torch.max(torch.abs(value)) > 0.5:
             print("INFO: Initial values seem somewhat large, but might be okay.")
        else:
             print("INFO: Initial values are small, initialization fix seems effective.")
        print(f"Policy output shape: {policy.shape}")
        # Policy sums might be slightly less than 1.0 if the mask removed all valid moves, or due to float precision
        print(f"Policy sums (should be ~1.0): {policy.sum(dim=1).detach().numpy()}")