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
    This version is simplified to be purely additive, which is a more standard approach.
    It learns embeddings for the absolute position of a square, its rank, its file,
    and its diagonals, then sums them up to create a final embedding.
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
        batch, seq_len, d = x.shape
        assert seq_len == 64, "This positional encoding is designed for an 8x8 board."

        # Create position indices (0..63)
        positions = torch.arange(seq_len, device=x.device)

        # Convert indices to 2D board coordinates.
        files = positions % 8
        ranks = positions // 8
        diagonals = ranks + files
        anti_diagonals = ranks - files + 7  # Shift to be non-negative

        # Get the embeddings for each component.
        file_emb = self.file_embedding(files)           # [64, d_model]
        rank_emb = self.rank_embedding(ranks)           # [64, d_model]
        diag_emb = self.diag_embedding(diagonals)       # [64, d_model]
        anti_diag_emb = self.anti_diag_embedding(anti_diagonals)  # [64, d_model]

        # Sum all positional components. The absolute embedding acts as a base.
        # Broadcast batch dimension automatically via unsqueeze.
        total_pos_embedding = (
            self.absolute_pos_embedding[:, :seq_len, :] +
            file_emb.unsqueeze(0) +
            rank_emb.unsqueeze(0) +
            diag_emb.unsqueeze(0) +
            anti_diag_emb.unsqueeze(0)
        )

        # Add the rich positional encoding to the input token embeddings.
        return x + total_pos_embedding


# ==============================================================================
# 2. Transformer Core Components
# These are the building blocks of the main network.
# ==============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention mechanism, but enhanced with the
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
        
        # 4. Apply mask if provided (e.g., for padding).
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
    Gated Linear Unit with GELU activation. This is a modern and effective
    replacement for the standard FFN in a transformer block. It often leads
    to better performance.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Project to twice the feed-forward dimension.
        self.proj = nn.Linear(d_model, d_ff * 2)
        # Project back to the model dimension.
        self.out = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Project and split into two parts.
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        # The "gate" (x1) multiplies the GELU-activated part (x2).
        return self.out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    """
    A single transformer block. This implementation uses Pre-LayerNorm for
    better training stability, and a GEGLU feed-forward network.
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
        # Pre-LN: Normalize -> Attention -> Dropout -> Residual Connection
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)
        
        # Pre-LN: Normalize -> FFN -> Dropout -> Residual Connection
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        return x


# ==============================================================================
# 3. Output Heads
# These modules produce the final policy and value outputs from the transformer's features.
# ==============================================================================

class PolicyHead(nn.Module):
    """
    Predicts the policy (a probability distribution over all possible moves).
    Outputs 72 move-type logits per square, flattened to 64*72 = 4608 classes
    to align with the project's move encoding and legal move mask.
    """
    def __init__(self, d_model, moves_per_square=72):
        super().__init__()
        # Project per-square features to 72 move-type logits.
        self.proj = nn.Linear(d_model, moves_per_square)

    def forward(self, x):
        # x: [batch, 64, d_model] -> logits: [batch, 64, 72]
        policy_logits = self.proj(x)
        # Flatten to [batch, 64*72]
        return policy_logits.reshape(x.shape[0], -1)


class ValueHead(nn.Module):
    """
    Predicts the value of the position (a single scalar from -1 to 1).
    It uses the special [CLS] token as a summary of the entire board state.
    """
    def __init__(self, d_model):
        super().__init__()
        self.value_proj = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid() # Sigmoid activation to scale the output to [0, 1] to match training targets
        )
        
    def forward(self, x):
        # The input x is the feature vector for the [CLS] token: [batch, 1, d_model]
        # We remove the sequence dimension before projecting.
        return self.value_proj(x.squeeze(1))


class WDLValueHead(nn.Module):
    """
    Win-Draw-Loss (WDL) value head. Predicts three probabilities: win, draw, and loss.
    This provides richer evaluation signal than a single scalar value and helps
    the model better understand complex endgame positions where draws are likely.
    """
    def __init__(self, d_model):
        super().__init__()
        self.value_proj = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),  # Output 3 logits for Win, Draw, Loss
        )
        
    def forward(self, x):
        # The input x is the feature vector for the [CLS] token: [batch, 1, d_model]
        # We remove the sequence dimension before projecting.
        return self.value_proj(x.squeeze(1))


# ==============================================================================
# 4. Main Model: Titan
# This class assembles all the components into the final network.
# ==============================================================================

class Titan(nn.Module):
    """
    A large, transformer-based neural network for chess, designed for high performance.

    Key Architectural Features:
    - Input Projection: A 1x1 convolution to create initial token embeddings from input planes.
    - CLS Token: A special token prepended to the sequence to aggregate global board information.
    - Rich Positional Encoding: A custom, additive positional encoding for chess geometry.
    - Transformer Backbone: A stack of transformer blocks using Pre-LayerNorm and GEGLU FFNs.
    - Relative Position Bias: Injects spatial awareness directly into the attention mechanism.
    - Decoupled Heads: Separate, simple heads for policy and value prediction.
    - Gradient Checkpointing: An option to save memory during training of this large model.
    """
    def __init__(
        self,
        num_layers=15,
        d_model=1024,
        num_heads=16,
        d_ff=4096,
        dropout=0.1,
        policy_weight=1.0,
        input_planes=112, # e.g., 14 piece types * 8 history + other features
        use_gradient_checkpointing=False,
        use_wdl=True,  # Use Win-Draw-Loss value head
    ):
        super().__init__()

        self.policy_weight = policy_weight
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_wdl = use_wdl

        # A 1x1 conv is an efficient way to project the input planes to the model's dimension.
        self.input_projection = nn.Conv2d(input_planes, d_model, kernel_size=1)
        
        # The special [CLS] token, whose final embedding will be used for the value head.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Our custom positional encoding module for the 64 board squares.
        self.positional_encoding = ChessPositionalEncoding(d_model)
        
        # The main body of the network: a stack of transformer blocks.
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final normalization layer.
        self.output_norm = nn.LayerNorm(d_model)

        # The output heads.
        if use_wdl:
            self.value_head = WDLValueHead(d_model)
        else:
            self.value_head = ValueHead(d_model)
        self.policy_head = PolicyHead(d_model)  # 64*72 = 4608 classes
        
        # Loss functions.
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters for better training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x, value_target=None, policy_target=None, policy_mask=None):
        batch_size = x.shape[0]
        
        # 1. Project input planes to token embeddings.
        x = self.input_projection(x)  # [batch, d_model, 8, 8]
        
        # 2. Reshape to a sequence of 64 tokens.
        x = x.flatten(2).transpose(1, 2)  # [batch, 64, d_model]
        
        # 3. Add chess-specific positional encodings to the board tokens.
        x = self.positional_encoding(x)
        
        # 4. Prepend the [CLS] token to the sequence.
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [batch, 65, d_model]
        
        # 5. Pass through the transformer blocks.
        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                # Gradient checkpointing saves memory at the cost of a bit of speed.
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        # 6. Apply final layer normalization.
        x = self.output_norm(x)

        # 7. Split the [CLS] token from the board tokens.
        cls_features = x[:, 0:1, :]   # [batch, 1, d_model]
        board_features = x[:, 1:, :] # [batch, 64, d_model]

        # 8. Compute value and policy from their respective features.
        value = self.value_head(cls_features)
        policy = self.policy_head(board_features)

        # --- Loss path when targets are provided ---
        if value_target is not None and policy_target is not None:
            if self.use_wdl:
                # Handle WDL value loss
                if value_target.dim() == 2 and value_target.shape[1] == 3:
                    # Already in WDL format
                    wdl_targets = value_target
                else:
                    # Convert scalar values to WDL (win/draw/loss)
                    # value_target is in [0, 1] range (0=loss, 0.5=draw, 1=win)
                    wdl_targets = torch.zeros(batch_size, 3, device=value_target.device)
                    
                    # Simple conversion: map [0, 1] to WDL probabilities
                    # Use a more sophisticated mapping that encourages draws in balanced positions
                    value_flat = value_target.view(-1)
                    
                    # Win probability: increases as value approaches 1
                    win_prob = torch.clamp(2 * value_flat - 1, 0, 1)  # Maps [0.5, 1] -> [0, 1]
                    # Loss probability: increases as value approaches 0  
                    loss_prob = torch.clamp(1 - 2 * value_flat, 0, 1)  # Maps [0, 0.5] -> [1, 0]
                    # Draw probability: peaks at value=0.5
                    draw_prob = 1 - torch.abs(2 * value_flat - 1)  # Maps to a peak at 0.5
                    
                    # Normalize to ensure probabilities sum to 1
                    total = win_prob + draw_prob + loss_prob + 1e-8
                    wdl_targets[:, 0] = win_prob / total
                    wdl_targets[:, 1] = draw_prob / total
                    wdl_targets[:, 2] = loss_prob / total
                
                # WDL loss using cross entropy with soft targets
                log_probs = F.log_softmax(value, dim=1)
                value_loss = -(wdl_targets * log_probs).sum(dim=1).mean()
            else:
                value_loss = self.mse_loss(value, value_target)

            # Optionally apply legal move mask to logits
            masked_policy = policy
            if policy_mask is not None:
                pm = policy_mask.view(batch_size, -1).float()
                masked_policy = masked_policy + (pm + 1e-8).log()  # avoid -inf in autograd

            # Handle index vs distribution targets
            if policy_target.dim() == 1 or (policy_target.dim() == 2 and policy_target.shape[1] == 1):
                policy_target = policy_target.view(batch_size).long()
                policy_loss = self.cross_entropy_loss(masked_policy, policy_target)
            else:
                # Distribution target (e.g., RL with soft targets)
                target = policy_target
                if policy_mask is not None:
                    pm = policy_mask.view(batch_size, -1).float()
                    target = target * pm
                target = target + 1e-8
                target = target / target.sum(dim=1, keepdim=True)
                log_probs = F.log_softmax(masked_policy, dim=1)
                policy_loss = -(target * log_probs).sum(dim=1).mean()

            total_loss = value_loss + self.policy_weight * policy_loss
            return total_loss, value_loss, policy_loss

        # --- Inference Path ---
        else:
            # During inference, we apply the legal move mask.
            if policy_mask is not None:
                policy_mask = policy_mask.view(batch_size, -1)
                # Set illegal move logits to a very small number.
                policy = policy.masked_fill(policy_mask == 0, -1e9)
            
            policy_softmax = F.softmax(policy, dim=1)
            
            # Convert WDL to scalar value for inference
            if self.use_wdl:
                # Convert WDL logits to scalar value: win_prob - loss_prob
                wdl_probs = F.softmax(value, dim=1)
                value_scalar = wdl_probs[:, 0:1] - wdl_probs[:, 2:3]  # Keep dimensions for compatibility
                return value_scalar, policy_softmax
            else:
                return value, policy_softmax


def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # --- Configuration ---
    config = {
        "num_layers": 15,
        "d_model": 1024,
        "num_heads": 16, # d_model must be divisible by num_heads
        "d_ff": 4096,
        "input_planes": 112,
        "dropout": 0.1
    }

    # --- Model Initialization ---
    model = Titan(**config)
    print(f"Titan model initialized.")
    print(f"Number of trainable parameters: {count_parameters(model):,}")
    
    # --- Test Forward Pass ---
    # Create dummy data that mimics a real use case.
    batch_size = 4
    dummy_input = torch.randn(batch_size, config["input_planes"], 8, 8)
    dummy_value_target = torch.randn(batch_size, 1)
    
    # Policy target is a single index per batch item representing the chosen move.
    num_total_moves = 72 * 64
    dummy_policy_target = torch.randint(0, num_total_moves, (batch_size, 1))
    
    # Policy mask indicates which moves are legal (1 for legal, 0 for illegal).
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
        # Check that the policy probabilities sum to 1 for each item in the batch.
        print(f"Policy sums (should be ~1.0): {policy.sum(dim=1).detach().numpy()}")
