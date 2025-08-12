import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. Positional Encoding Modules
# These modules are crucial for giving the transformer a sense of the 8x8 board.
# ==============================================================================

class RelativePositionBias(nn.Module):
    """
    Computes a learnable bias for the attention mechanism based on the relative
    position of the query and key. This is a key improvement for transformers
    in spatial domains like chess, as it directly teaches the model about
    the geometry of the board (e.g., how a square 2 ranks away and 1 file
    away relates to the current square).
    """
    def __init__(self, num_heads, max_relative_position=7):
        super().__init__()
        self.num_heads = num_heads
        self.max_relative_position = max_relative_position
        # We need a bucket for each relative position from -7 to +7, so 15 buckets.
        num_buckets = 2 * max_relative_position + 1
        
        # This table stores the learnable biases.
        # It's a 3D tensor: (rel_rank_bucket, rel_file_bucket, num_heads)
        self.relative_bias_table = nn.Parameter(
            torch.zeros(num_buckets, num_buckets, num_heads)
        )
        # Initialize the biases to be small random values.
        nn.init.normal_(self.relative_bias_table, std=0.02)

    def forward(self, seq_len):
        """
        Generates the bias tensor for a given sequence length. Supports 64
        (board tokens only) or 65 (CLS + 64 board tokens). For seq_len=65,
        CLS interactions receive zero bias while board-board entries receive
        the learned relative positional bias.
        """
        assert seq_len in (64, 65)

        board_len = 64
        # Create coordinates for each square on the board (0..63)
        positions = torch.arange(board_len, device=self.relative_bias_table.device)
        ranks = positions // 8
        files = positions % 8
        
        # Calculate the relative distance between every pair of squares.
        # The result is a (seq_len, seq_len) matrix for ranks and files.
        relative_ranks = ranks.unsqueeze(1) - ranks.unsqueeze(0)
        relative_files = files.unsqueeze(1) - files.unsqueeze(0)
        
        # Clip the distances to be within our max_relative_position range.
        clipped_rel_ranks = relative_ranks.clamp(-self.max_relative_position, self.max_relative_position)
        clipped_rel_files = relative_files.clamp(-self.max_relative_position, self.max_relative_position)
        
        # Shift the indices to be positive for table lookup (0 to 14).
        rank_indices = clipped_rel_ranks + self.max_relative_position
        file_indices = clipped_rel_files + self.max_relative_position
        
        # Look up the bias from the learnable table.
        bias = self.relative_bias_table[rank_indices, file_indices]
        
        # Reshape to (1, num_heads, 64, 64)
        bias_64 = bias.permute(2, 0, 1).unsqueeze(0)

        if seq_len == 64:
            return bias_64

        # For 65 (CLS + 64), pad zeros for CLS interactions
        device = bias_64.device
        out = torch.zeros(1, self.num_heads, 65, 65, device=device, dtype=bias_64.dtype)
        out[:, :, 1:, 1:] = bias_64
        return out


class ChessPositionalEncoding(nn.Module):
    """
    Creates a rich, chess-specific positional embedding for each square on the board.
    This version is purely additive, which is a standard and effective approach.
    It learns embeddings for the absolute position, rank, file, and diagonals,
    then sums them up to create a final, comprehensive positional embedding.
    """
    def __init__(self, d_model, max_seq_len=64):
        super().__init__()
        # Main learnable embedding for each of the 64 squares.
        self.absolute_pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Learnable embeddings for each of the 8 files and 8 ranks.
        self.file_embedding = nn.Embedding(8, d_model)
        self.rank_embedding = nn.Embedding(8, d_model)
        
        # Learnable embeddings for the 15 diagonals and 15 anti-diagonals.
        self.diag_embedding = nn.Embedding(15, d_model)
        self.anti_diag_embedding = nn.Embedding(15, d_model)

    def forward(self, x):
        """
        Adds the combined positional encodings to the input tensor `x`.
        """
        seq_len = x.shape[1]
        assert seq_len == 64, "This positional encoding is designed for an 8x8 board."

        # Create position indices (0..63)
        positions = torch.arange(seq_len, device=x.device)
        
        # Convert indices to 2D board coordinates.
        files = positions % 8
        ranks = positions // 8
        diagonals = ranks + files
        anti_diagonals = ranks - files + 7 # Shift to be non-negative

        # Get the embeddings for each component.
        file_emb = self.file_embedding(files)
        rank_emb = self.rank_embedding(ranks)
        diag_emb = self.diag_embedding(diagonals)
        anti_diag_emb = self.anti_diag_embedding(anti_diagonals)

        # Sum all positional components. The absolute embedding acts as a base.
        # Broadcasting automatically handles the batch dimension.
        total_pos_embedding = (
            self.absolute_pos_embedding[:, :seq_len, :] + 
            file_emb + 
            rank_emb + 
            diag_emb + 
            anti_diag_emb
        )
        
        # Add the rich positional encoding to the input token embeddings.
        return x + total_pos_embedding


# ==============================================================================
# 2. Transformer Core Components
# These are the building blocks of the main network.
# ==============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention mechanism, enhanced with the
    chess-specific RelativePositionBias.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Query, Key, Value, and Output.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.relative_position_bias = RelativePositionBias(num_heads)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 1. Project and reshape Q, K, V into multiple heads.
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Compute attention scores (scaled dot-product).
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. Add the learnable relative position bias for board tokens.
        if seq_len in (64, 65):
            scores = scores + self.relative_position_bias(seq_len)
        
        # 4. Apply mask if provided.
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 5. Apply softmax and dropout.
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 6. Compute the weighted sum of values.
        context = torch.matmul(attention_weights, V)
        
        # 7. Concatenate heads and apply final linear layer.
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output


class GEGLU(nn.Module):
    """
    Gated Linear Unit with GELU activation. A modern and effective
    replacement for the standard FFN in a transformer block.
    """
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
    A single transformer block using Pre-LayerNorm for stability
    and a GEGLU feed-forward network.
    """
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
    """
    Predicts the policy (a probability distribution over all possible moves).
    """
    def __init__(self, d_model, num_move_actions_per_square=72):
        super().__init__()
        self.num_move_actions_per_square = num_move_actions_per_square
        self.proj = nn.Linear(d_model, num_move_actions_per_square)

    def forward(self, x):
        # Input x shape: [batch, 64, d_model]
        # Project features for each of the 64 squares.
        policy_logits = self.proj(x)  # [batch, 64, num_move_actions_per_square]
        # Reshape to a flat policy vector: [batch, 64 * num_move_actions_per_square]
        return policy_logits.view(x.shape[0], -1)


class ValueHead(nn.Module):
    """
    Predicts the value of the position using the special [CLS] token.
    Compatible with both old (2-layer) and new (3-layer) architectures.
    Now uses Tanh activation (like AlphaZero) for better gradient flow.
    """
    def __init__(self, d_model, legacy_mode=False):
        super().__init__()
        self.legacy_mode = legacy_mode
        self.d_model = d_model
        
        if legacy_mode:
            # Legacy architecture with 2 layers
            # Now using Tanh for better training stability
            self.fc1 = nn.Linear(d_model, d_model // 2)
            self.activation1 = nn.GELU()
            self.dropout1 = nn.Dropout(0.1)
            self.fc2 = nn.Linear(d_model // 2, 1)
            self.output_activation = nn.Tanh()  # Changed from Sigmoid to Tanh
            
            # Initialize with Xavier (optimal for Tanh)
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            # Smaller initialization for final layer to prevent saturation
            nn.init.xavier_normal_(self.fc2.weight, gain=0.1)
            nn.init.zeros_(self.fc2.bias)  # Critical: start with 0 bias
        else:
            # New architecture with 3 layers
            self.fc1 = nn.Linear(d_model, d_model // 2)
            self.activation1 = nn.GELU()
            self.dropout1 = nn.Dropout(0.1)
            self.fc2 = nn.Linear(d_model // 2, 128)
            self.activation2 = nn.GELU()
            self.dropout2 = nn.Dropout(0.1)
            self.fc3 = nn.Linear(128, 1)
            self.output_activation = nn.Tanh()  # Changed from Sigmoid to Tanh
            
            # Initialize with Xavier (optimal for Tanh)
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
            # Smaller initialization for final layer
            nn.init.xavier_normal_(self.fc3.weight, gain=0.1)
            nn.init.zeros_(self.fc3.bias)  # Critical: start with 0 bias
        
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
    This provides richer evaluation signal than a single scalar value and helps
    the model better understand complex endgame positions where draws are likely.
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
# 4. Main Model: Titan-Mini
# ==============================================================================

class TitanMini(nn.Module):
    """
    A smaller, CPU-friendly transformer-based neural network for chess.
    Designed to be under 100MB (<25M parameters) for deployment on low-end hardware.

    Key Architectural Features:
    - Reduced dimensions (d_model, d_ff) and layers for efficiency.
    - Retains the powerful features of the larger Titan model:
        - Input Projection: A 1x1 convolution for token embeddings.
        - CLS Token: For global board state aggregation.
        - Rich Positional Encoding: Custom chess-specific embeddings.
        - Transformer Backbone: Stack of Pre-LN, GEGLU transformer blocks.
        - Relative Position Bias: Injects spatial awareness into attention.
    """
    def __init__(
        self,
        num_layers=13,  # Updated default for 200MB model
        d_model=512,    # Updated default for 200MB model
        num_heads=8,    # Updated default for 200MB model
        d_ff=1920,      # Updated default for 200MB model (3.75x d_model)
        dropout=0.1,
        policy_weight=1.0,
        input_planes=112,
        use_wdl=True,  # Use Win-Draw-Loss value head
        legacy_value_head=False,  # Use legacy 2-layer value head for old models
    ):
        super().__init__()

        self.policy_weight = policy_weight
        self.d_model = d_model
        self.use_wdl = use_wdl

        # Project input planes to the model's embedding dimension.
        self.input_projection = nn.Conv2d(input_planes, d_model, kernel_size=1)
        
        # The special [CLS] token for the value head.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Custom positional encoding for the 64 board squares.
        self.positional_encoding = ChessPositionalEncoding(d_model)
        
        # The main body of the network.
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final normalization layer.
        self.output_norm = nn.LayerNorm(d_model)

        # Output heads. Policy uses 72 move types per square (standard for this codebase).
        if use_wdl:
            self.value_head = WDLValueHead(d_model)
        else:
            self.value_head = ValueHead(d_model, legacy_mode=legacy_value_head)
        self.policy_head = PolicyHead(d_model, num_move_actions_per_square=72)
        
        # Loss functions.
        self.mse_loss = nn.MSELoss()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()  # For legacy mode
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters for better training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x, value_target=None, policy_target=None, policy_mask=None, 
                valueTarget=None, policyTarget=None, policyMask=None):
        # Handle both camelCase and snake_case parameter names for compatibility
        if valueTarget is not None:
            value_target = valueTarget
        if policyTarget is not None:
            policy_target = policyTarget
        if policyMask is not None:
            policy_mask = policyMask
            
        batch_size = x.shape[0]
        
        # 1. Project input planes to token embeddings.
        x = self.input_projection(x)  # [batch, d_model, 8, 8]
        
        # 2. Reshape to a sequence of 64 tokens.
        x = x.flatten(2).transpose(1, 2)  # [batch, 64, d_model]
        
        # 3. Add chess-specific positional encodings.
        x = self.positional_encoding(x)
        
        # 4. Prepend the [CLS] token.
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [batch, 65, d_model]
        
        # 5. Pass through the transformer blocks.
        for block in self.transformer_blocks:
            x = block(x)
        
        # 6. Apply final layer normalization.
        x = self.output_norm(x)

        # 7. Split features for the two heads.
        cls_features = x[:, 0:1, :]   # [batch, 1, d_model]
        board_features = x[:, 1:, :] # [batch, 64, d_model]

        # 8. Compute value and policy.
        value = self.value_head(cls_features)
        policy = self.policy_head(board_features)
        
        # --- Training Path ---
        if self.training:
            assert value_target is not None and policy_target is not None
            if self.use_wdl:
                # Handle WDL value loss
                if value_target.dim() == 2 and value_target.shape[1] == 3:
                    # Already in WDL format
                    wdl_targets = value_target
                else:
                    # Convert scalar values to WDL (win/draw/loss)
                    # value_target is in [-1, 1] range (-1=loss, 0=draw, 1=win)
                    wdl_targets = torch.zeros(batch_size, 3, device=value_target.device)
                    
                    # IMPROVED WDL CONVERSION: Better mapping from scalar to WDL probabilities
                    # This uses a softer conversion that better represents uncertainty
                    value_flat = value_target.view(-1)
                    
                    # Use a temperature parameter to control sharpness
                    temperature = 0.5  # Lower = sharper peaks, Higher = more uniform
                    
                    # Define target values for each outcome (now in [-1, 1] range)
                    win_value = 1.0   # Win = 1
                    draw_value = 0.0  # Draw = 0
                    loss_value = -1.0 # Loss = -1
                    
                    # Compute distances from value to each outcome
                    dist_to_win = torch.abs(value_flat - win_value)
                    dist_to_draw = torch.abs(value_flat - draw_value)
                    dist_to_loss = torch.abs(value_flat - loss_value)
                    
                    # Convert distances to probabilities using softmax-like approach
                    win_logit = -dist_to_win / temperature
                    draw_logit = -dist_to_draw / temperature
                    loss_logit = -dist_to_loss / temperature
                    
                    # Stack and apply softmax to get probabilities
                    logits = torch.stack([win_logit, draw_logit, loss_logit], dim=1)
                    wdl_probs = F.softmax(logits, dim=1)
                    
                    wdl_targets[:, 0] = wdl_probs[:, 0]  # Win probability
                    wdl_targets[:, 1] = wdl_probs[:, 1]  # Draw probability
                    wdl_targets[:, 2] = wdl_probs[:, 2]  # Loss probability
                
                # WDL loss using cross entropy with soft targets
                log_probs = F.log_softmax(value, dim=1)
                value_loss = -(wdl_targets * log_probs).sum(dim=1).mean()
            else:
                # For non-WDL mode, use MSE loss
                value_loss = self.mse_loss(value, value_target)
            
            # Handle both soft targets (distribution) and hard targets (single index)
            if policy_target.dim() == 1 or (policy_target.dim() == 2 and policy_target.size(1) == 1):
                # Hard targets: single index per sample
                policy_target_1d = policy_target.view(batch_size)
                policy_loss = self.cross_entropy_loss(policy, policy_target_1d)
            else:
                # Soft targets: full distribution
                policy_target_flat = policy_target.view(batch_size, -1)
                # Use KL divergence for soft targets
                log_policy = F.log_softmax(policy, dim=1)
                policy_loss = F.kl_div(log_policy, policy_target_flat, reduction='batchmean')
            
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
                # Convert WDL logits to scalar value: win_prob - loss_prob
                wdl_probs = F.softmax(value, dim=1)
                value_scalar = wdl_probs[:, 0:1] - wdl_probs[:, 2:3]  # Keep dimensions for compatibility
                return value_scalar, policy_softmax
            else:
                # Non-WDL mode: value already has Tanh applied (outputs in [-1, 1])
                return value, policy_softmax


def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # --- Model Initialization ---
    model = TitanMini()
    num_params = count_parameters(model)
    model_size_mb = num_params * 4 / (1024 * 1024) # 4 bytes per float32 parameter

    print(f"Titan-Mini model initialized.")
    print(f"Number of trainable parameters: {num_params:,}")
    print(f"Estimated model size: {model_size_mb:.2f} MB")
    
    # --- Test Forward Pass ---
    batch_size = 4
    dummy_input = torch.randn(batch_size, 112, 8, 8)
    dummy_value_target = torch.randn(batch_size, 1)
    
    num_total_moves = 72 * 64
    dummy_policy_target = torch.randint(0, num_total_moves, (batch_size, 1))
    dummy_mask = torch.randint(0, 2, (batch_size, num_total_moves))

    # --- Training Mode Test ---
    print("\n--- Testing Training Mode ---")
    model.train()
    total_loss, value_loss, policy_loss = model(
        dummy_input, 
        value_target=dummy_value_target,
        policy_target=dummy_policy_target,
    )
    print(f"Total loss: {total_loss.item():.4f}, Value loss: {value_loss.item():.4f}, Policy loss: {policy_loss.item():.4f}")
    
    # --- Inference Mode Test ---
    print("\n--- Testing Inference Mode ---")
    model.eval()
    with torch.no_grad():
        value, policy = model(dummy_input, policy_mask=dummy_mask)
        print(f"Value output shape: {value.shape}")
        print(f"Policy output shape: {policy.shape}")
        print(f"Policy sums (should be ~1.0): {policy.sum(dim=1).detach().numpy()}")

