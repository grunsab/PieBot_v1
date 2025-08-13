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
    def __init__(self, d_model, max_seq_len=64):
        super().__init__()
        # Initialize parameters to zeros; actual initialization happens in _init_parameters
        self.absolute_pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
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
    # (Implementation remains the same)
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
# 3. Output Heads
# ==============================================================================

class PolicyHead(nn.Module):
    def __init__(self, d_model, num_move_actions_per_square=72):
        super().__init__()
        self.num_move_actions_per_square = num_move_actions_per_square
        self.proj = nn.Linear(d_model, num_move_actions_per_square)

    def forward(self, x):
        policy_logits = self.proj(x)
        return policy_logits.view(x.shape[0], -1)


class ValueHead(nn.Module):
    def __init__(self, d_model, legacy_mode=False):
        super().__init__()
        self.legacy_mode = legacy_mode
        self.d_model = d_model
        
        if legacy_mode:
            self.fc1 = nn.Linear(d_model, d_model // 2)
            self.activation1 = nn.GELU()
            self.dropout1 = nn.Dropout(0.1)
            self.fc2 = nn.Linear(d_model // 2, 1)
            self.output_activation = nn.Tanh()
        else:
            self.fc1 = nn.Linear(d_model, d_model // 2)
            self.activation1 = nn.GELU()
            self.dropout1 = nn.Dropout(0.1)
            self.fc2 = nn.Linear(d_model // 2, 128)
            self.activation2 = nn.GELU()
            self.dropout2 = nn.Dropout(0.1)
            self.fc3 = nn.Linear(128, 1)
            self.output_activation = nn.Tanh()
        
    def forward(self, x):
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
    def __init__(self, d_model):
        super().__init__()
        self.value_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),
        )
        
    def forward(self, x):
        return self.value_proj(x.squeeze(1))


# ==============================================================================
# 4. Main Model: Titan-Mini (Major Update: Adaptive Input Handling)
# ==============================================================================

class TitanMini(nn.Module):
    """
    Titan-Mini Transformer network adapted for both dense (112-plane) and 
    sparse (16-plane) input representations.
    """
    def __init__(
        self,
        num_layers=13,
        d_model=512,
        num_heads=8,
        d_ff=1920,
        dropout=0.1,
        policy_weight=1.0,
        input_planes=112, # Handles both 112 (default) and 16
        use_wdl=True,
        legacy_value_head=False,
    ):
        super().__init__()

        self.policy_weight = policy_weight
        self.d_model = d_model
        self.use_wdl = use_wdl
        self.input_planes = input_planes

        # --- FIX: Adaptive Input Handling Strategy ---
        # If input planes <= 16, assume sparse encoding.
        self.use_piece_embeddings = (input_planes <= 16)
        
        # Initialize placeholders
        self.input_projection = None
        self.piece_embedding = None
        self.cls_token = None
        self.cls_projection = None

        if self.use_piece_embeddings:
            # Strategy for Sparse Inputs (e.g., 16 planes)
            if input_planes < 12:
                 raise ValueError(f"Sparse input mode requires at least 12 planes for pieces, got {input_planes}")

            # 1. Piece Embeddings (13: 1 empty + 12 pieces)
            self.piece_embedding = nn.Embedding(13, d_model)
            
            # 2. CLS Token derived from Global Features (Planes 12+)
            num_global_features = input_planes - 12
            if num_global_features > 0:
                self.cls_projection = nn.Linear(num_global_features, d_model)
            else:
                # Fallback if exactly 12 planes are used
                self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        else:
            # Strategy for Dense Inputs (e.g., 112 planes)
            # 1. 1x1 Convolution Projection
            self.input_projection = nn.Conv2d(input_planes, d_model, kernel_size=1)
            # 2. Learned CLS Token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # --- End of Input Handling Strategy ---

        # Shared components
        self.positional_encoding = ChessPositionalEncoding(d_model)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(d_model)

        # Output heads
        if use_wdl:
            self.value_head = WDLValueHead(d_model)
        else:
            self.value_head = ValueHead(d_model, legacy_mode=legacy_value_head)
        self.policy_head = PolicyHead(d_model, num_move_actions_per_square=72)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # Initialize parameters
        self._init_parameters()
        self._init_head_specific_weights()

    def _planes_to_indices(self, x):
        """
        Converts sparse one-hot encoded piece planes (first 12 planes) into indices.
        x shape: [B, C, 8, 8]
        Returns: [B, 64] tensor of indices (0=Empty, 1-12=Pieces)
        """
        B, C, H, W = x.shape
        
        # Select the first 12 planes and flatten the spatial dimensions
        piece_planes = x[:, :12, :, :].view(B, 12, H*W)
        
        # Find the index of the active plane (0-11)
        indices = torch.argmax(piece_planes, dim=1) # [B, H*W]
        
        # Check if the square is actually occupied (sum > 0.5)
        is_occupied = torch.sum(piece_planes, dim=1) > 0.5 # [B, H*W]
        
        # If occupied, index is 1-based (1 to 12). If empty, index is 0.
        piece_indices = (indices + 1) * is_occupied.long() # [B, 64]
        
        return piece_indices
        
    def _init_parameters(self):
        """Initialize parameters. Standardized approach for stability."""
        
        # 1. Generic Initialization (Xavier Uniform for weights, Zeros for biases)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 2. Embedding Initialization (Normal distribution, std=0.02)
        # Crucial for transformer stability.
        
        # Positional Embeddings
        if hasattr(self.positional_encoding, 'absolute_pos_embedding'):
             nn.init.normal_(self.positional_encoding.absolute_pos_embedding, std=0.02)

        for m in self.positional_encoding.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        
        # Piece Embeddings (if used)
        if self.piece_embedding is not None:
            nn.init.normal_(self.piece_embedding.weight, std=0.02)

        # 3. CLS Token Initialization (if used)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        

    def _init_head_specific_weights(self):
        # (Implementation remains the same as the previously corrected version, ensuring small gain on final layers)
        """
        Applies specific initialization for the policy and value heads (small gain).
        """
        
        # 1. Policy Head Initialization
        if hasattr(self.policy_head, 'proj') and isinstance(self.policy_head.proj, nn.Linear):
            nn.init.xavier_uniform_(self.policy_head.proj.weight, gain=0.1)
            if self.policy_head.proj.bias is not None:
                nn.init.zeros_(self.policy_head.proj.bias)

        # 2. Value Head Initialization
        if self.use_wdl:
            if hasattr(self.value_head, 'value_proj') and isinstance(self.value_head.value_proj, nn.Sequential):
                final_layer = self.value_head.value_proj[-1]
                if isinstance(final_layer, nn.Linear):
                    nn.init.xavier_uniform_(final_layer.weight, gain=0.1)

        else:
            # Standard ValueHead (Tanh output)
            final_layer = None
            if hasattr(self.value_head, 'legacy_mode'):
                if self.value_head.legacy_mode and hasattr(self.value_head, 'fc2'):
                    final_layer = self.value_head.fc2
                elif not self.value_head.legacy_mode and hasattr(self.value_head, 'fc3'):
                    final_layer = self.value_head.fc3
            
            if final_layer is not None:
                # Use xavier_normal_ for Tanh output layers
                nn.init.xavier_normal_(final_layer.weight, gain=0.1)

    
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
        
        # 1. Input Processing (Adaptive Strategy)
        if self.use_piece_embeddings:
            # --- Sparse Input Path (e.g., 16 planes) ---
            
            # 1a. Create CLS token
            if self.input_planes > 12 and self.cls_projection is not None:
                # Global features (Planes 12+) are assumed constant across the board; take from (0,0).
                global_features = x[:, 12:, 0, 0] # [B, num_global_features]
                cls = self.cls_projection(global_features).unsqueeze(1) # [B, 1, d_model]
            elif self.cls_token is not None:
                cls = self.cls_token.expand(batch_size, -1, -1)
            else:
                 raise RuntimeError("Configuration error: CLS token generation failed in sparse mode.")

            # 1b. Convert piece planes (0-11) to indices and get embeddings
            piece_indices = self._planes_to_indices(x) # [B, 64]
            board_tokens = self.piece_embedding(piece_indices) # [B, 64, d_model]

        else:
            # --- Dense Input Path (e.g., 112 planes) ---

            # 1a. Project input planes using 1x1 Conv
            board_tokens = self.input_projection(x)  # [B, d_model, 8, 8]
            
            # 1b. Reshape to sequence [B, 64, d_model]
            board_tokens = board_tokens.flatten(2).transpose(1, 2)
            
            # 1c. Use learned CLS token
            cls = self.cls_token.expand(batch_size, -1, -1)

        
        # 2. Add positional encodings to board tokens.
        board_tokens = self.positional_encoding(board_tokens)
        
        # 3. Combine CLS token and board tokens.
        x_combined = torch.cat([cls, board_tokens], dim=1)  # [B, 65, d_model]
        
        # 4. Transformer blocks.
        for block in self.transformer_blocks:
            x_combined = block(x_combined)
        
        # 5. Final normalization.
        x_combined = self.output_norm(x_combined)

        # 6. Split features.
        cls_features = x_combined[:, 0:1, :]   # [B, 1, d_model]
        board_features = x_combined[:, 1:, :] # [B, 64, d_model]

        # 7. Compute value and policy.
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
                    W = torch.relu(value_flat)
                    L = torch.relu(-value_flat)
                    D = 1.0 - W - L
                    D = torch.clamp(D, min=0.0)
                    # Assumes WDLValueHead output order is [Win, Draw, Loss]
                    wdl_targets = torch.stack([W, D, L], dim=1)

                log_probs = F.log_softmax(value, dim=1)
                # Ensure normalization
                wdl_targets = wdl_targets / (wdl_targets.sum(dim=1, keepdim=True) + 1e-8)
                # Cross-entropy
                value_loss = -(wdl_targets * log_probs).sum(dim=1).mean()
            else:
                value_loss = self.mse_loss(value, value_target)
            
            # Policy Loss Calculation (Cross-Entropy for soft targets)
            if policy_target.dim() == 1 or (policy_target.dim() == 2 and policy_target.size(1) == 1):
                # Hard targets
                policy_target_1d = policy_target.view(batch_size)
                policy_loss = self.cross_entropy_loss(policy, policy_target_1d)
            else:
                # Soft targets
                policy_target_flat = policy_target.view(batch_size, -1)
                log_policy = F.log_softmax(policy, dim=1)
                
                # Normalize targets
                policy_target_flat = policy_target_flat + 1e-8
                policy_target_flat = policy_target_flat / policy_target_flat.sum(dim=1, keepdim=True)

                # Cross-entropy
                policy_loss = -(policy_target_flat * log_policy).sum(dim=1).mean()
            
            total_loss = value_loss + self.policy_weight * policy_loss
            return total_loss, value_loss, policy_loss
        
        # --- Inference Path ---
        else:
            if policy_mask is not None:
                policy_mask = policy_mask.view(batch_size, -1)
                policy = policy.masked_fill(policy_mask == 0, -1e9)
            
            policy_softmax = F.softmax(policy, dim=1)
            
            if self.use_wdl:
                # E[V] = P(W) - P(L)
                wdl_probs = F.softmax(value, dim=1)
                value_scalar = wdl_probs[:, 0:1] - wdl_probs[:, 2:3]
                return value_scalar, policy_softmax
            else:
                return value, policy_softmax


def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # --- Model Initialization Test (Sparse Mode - Used in the training script) ---
    print("--- Testing TitanMini (Sparse 16-plane Mode) ---")
    model_sparse = TitanMini(use_wdl=True, input_planes=16)

    print(f"Using Piece Embeddings: {model_sparse.use_piece_embeddings}")
    print(f"Number of parameters: {count_parameters(model_sparse):,}")
    
    # --- Test Forward Pass (Sparse Mode) ---
    batch_size = 2
    dummy_input_sparse = torch.zeros(batch_size, 16, 8, 8)
    
    # Simulate placing pieces (Plane 0=WPawn, Plane 7=BKnight)
    dummy_input_sparse[0, 0, 4, 4] = 1 # WPawn on (4,4)
    dummy_input_sparse[0, 7, 2, 5] = 1 # BKnight on (2,5)
    # Simulate global features (Planes 12-15)
    dummy_input_sparse[:, 12:, :, :] = torch.rand(batch_size, 4, 8, 8)

    # Test targets
    dummy_value_target = torch.tensor([[0.5], [-0.2]])
    num_total_moves = 72 * 64
    dummy_policy_target = torch.rand(batch_size, num_total_moves)
    dummy_policy_target /= dummy_policy_target.sum(dim=1, keepdim=True)
    
    # Training Mode Test
    model_sparse.train()
    total_loss, value_loss, policy_loss = model_sparse(
        dummy_input_sparse, 
        value_target=dummy_value_target,
        policy_target=dummy_policy_target,
    )
    print(f"Sparse Mode Loss: {total_loss.item():.4f}")

    # --- Model Initialization Test (Dense Mode) ---
    print("\n--- Testing TitanMini (Dense 112-plane Mode) ---")
    model_dense = TitanMini(use_wdl=True, input_planes=112)
    print(f"Using Piece Embeddings: {model_dense.use_piece_embeddings}")
    
    # Test Forward Pass (Dense Mode)
    dummy_input_dense = torch.randn(batch_size, 112, 8, 8)
    model_dense.train()
    total_loss, _, _ = model_dense(
        dummy_input_dense,
        value_target=dummy_value_target,
        policy_target=dummy_policy_target,
    )
    print(f"Dense Mode Loss: {total_loss.item():.4f}")