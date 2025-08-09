import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from positional_encoding import RoPE2D, ALiBi2D


class ChessPositionalEncoding(nn.Module):
    """
    Chess-specific positional encoding for transformer.
    Encodes spatial relationships between chess squares.
    """

    def __init__(self, d_model, max_seq_len=64):
        super().__init__()
        self.d_model = d_model

        # Learnable positional embeddings for each token (64 board squares)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # Additional chess-specific encodings
        part = max(1, d_model // 4)
        self.file_embedding = nn.Embedding(8, part)
        self.rank_embedding = nn.Embedding(8, part)
        self.diagonal_embedding = nn.Embedding(15, part)  # 15 diagonals
        self.anti_diagonal_embedding = nn.Embedding(15, part)

        # Projection to combine all encodings
        self.projection = nn.Linear(d_model + 4 * part, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Generate position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # Convert linear positions to 2D chess coordinates
        files = positions % 8
        ranks = positions // 8

        # Calculate diagonal indices (0-14)
        diagonals = ranks + files
        anti_diagonals = ranks - files + 7

        # Get embeddings (clamp indices to valid ranges)
        file_emb = self.file_embedding(files.clamp(0, 7))
        rank_emb = self.rank_embedding(ranks.clamp(0, 7))
        diag_emb = self.diagonal_embedding(diagonals.clamp(0, 14))
        anti_diag_emb = self.anti_diagonal_embedding(anti_diagonals.clamp(0, 14))

        # Concatenate chess-specific encodings
        chess_encoding = torch.cat([file_emb, rank_emb, diag_emb, anti_diag_emb], dim=-1)

        # Combine with learnable positional embedding
        combined = torch.cat([x + self.pos_embedding[:, :seq_len, :], chess_encoding], dim=-1)

        return self.projection(combined)


class RelativePositionBias(nn.Module):
    """
    Relative position bias for attention mechanism.
    Inspired by Shaw et al. and adapted for chess.
    """
    
    def __init__(self, num_heads, max_relative_position=7):
        super().__init__()
        self.num_heads = num_heads
        self.max_relative_position = max_relative_position
        
        # Learnable biases for relative positions
        num_buckets = 2 * max_relative_position + 1
        self.relative_bias_table = nn.Parameter(
            torch.zeros(num_buckets, num_buckets, num_heads)
        )
        
    def forward(self, seq_len):
        # Generate relative position matrix
        positions = torch.arange(seq_len)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # Convert to chess board coordinates
        from_files = positions % 8
        from_ranks = positions // 8
        to_files = positions % 8
        to_ranks = positions // 8
        
        # Calculate relative file and rank distances
        rel_files = from_files.unsqueeze(1) - to_files.unsqueeze(0)
        rel_ranks = from_ranks.unsqueeze(1) - to_ranks.unsqueeze(0)
        
        # Clip to maximum relative position
        rel_files = rel_files.clamp(-self.max_relative_position, self.max_relative_position)
        rel_ranks = rel_ranks.clamp(-self.max_relative_position, self.max_relative_position)
        
        # Convert to indices
        rel_file_idx = rel_files + self.max_relative_position
        rel_rank_idx = rel_ranks + self.max_relative_position
        
        # Get bias values
        bias = self.relative_bias_table[rel_file_idx, rel_rank_idx]
        
        return bias.permute(2, 0, 1).unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with relative position bias.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1, use_rope=False, use_alibi=False):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        self.use_alibi = use_alibi
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.relative_position_bias = RelativePositionBias(num_heads)
        # Optional 2D RoPE and ALiBi
        self.rope = RoPE2D(self.d_k) if use_rope else None
        self.alibi = ALiBi2D(num_heads) if use_alibi else None
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply 2D RoPE to Q and K if enabled
        if self.rope is not None:
            # If a leading CLS token is present (seq_len = 65 for 8x8 board + CLS),
            # apply RoPE only to the 64 board tokens to avoid out-of-range indices.
            if x.size(1) == 65:
                q_cls, q_board = Q[:, :, :1, :], Q[:, :, 1:, :]
                k_cls, k_board = K[:, :, :1, :], K[:, :, 1:, :]
                q_board, k_board = self.rope(q_board, k_board)
                Q = torch.cat([q_cls, q_board], dim=2)
                K = torch.cat([k_cls, k_board], dim=2)
            else:
                Q, K = self.rope(Q, K)

        # Scaled dot-product attention with relative position bias
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position bias
        position_bias = self.relative_position_bias(seq_len).to(x.device)
        scores = scores + position_bias
        # Add ALiBi bias if enabled
        if self.alibi is not None:
            scores = scores + self.alibi(seq_len).to(x.device)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.W_o(context)
        
        return output


class GEGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.proj = nn.Linear(d_model, d_ff * 2)
        self.out = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return self.out(F.gelu(x2) * x1)


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with Pre-LayerNorm and GEGLU FFN.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_rope=False, use_alibi=False):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout, use_rope=use_rope, use_alibi=use_alibi)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = GEGLU(d_model, d_ff)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN attention
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)
        # Pre-LN FFN
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        return x


class SmolgenInspiredProjection(nn.Module):
    """
    Projection layer inspired by Leela's smolgen mechanism.
    Enhances positional information for attention.
    """
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Initial projection to smaller dimension
        self.initial_proj = nn.Linear(d_model, 256)
        
        # Spatial mixing
        self.spatial_mix = nn.Linear(64 * 32, 256)
        
        # Per-head projections
        self.head_projections = nn.ModuleList([
            nn.Linear(256, d_model // num_heads) for _ in range(num_heads)
        ])
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Initial projection
        proj = self.initial_proj(x)  # [batch, seq_len, 256]

        # Reshape for spatial mixing (avoid non-contiguity issues)
        proj_flat = proj[..., :32]  # [batch, seq_len, 32]
        proj_spatial = proj_flat.reshape(batch_size, -1)  # [batch, seq_len*32]

        # Apply spatial mixing
        spatial_features = self.spatial_mix(proj_spatial)  # [batch, 256]

        # Generate per-head features
        head_features = []
        for head_proj in self.head_projections:
            head_feat = head_proj(spatial_features)  # [batch, d_model/num_heads]
            head_features.append(head_feat)

        # Stack and reshape
        head_features = torch.stack(head_features, dim=1)  # [batch, num_heads, d_model/num_heads]
        head_features = head_features.unsqueeze(2).expand(-1, -1, seq_len, -1)

        return head_features.reshape(batch_size, seq_len, self.d_model)


class PolicyHead(nn.Module):
    """
    Attention-based policy head for move prediction.
    Outputs a 64x64 matrix representing all possible moves.
    """
    
    def __init__(self, d_model):
        super().__init__()
        
        # Project to query/key dimensions
        self.query_proj = nn.Linear(d_model, 256)
        self.key_proj = nn.Linear(d_model, 256)
        
        # Additional context projection
        self.context_proj = nn.Linear(d_model, 128)
        
        # Final projection to policy logits
        self.policy_proj = nn.Linear(128, 72)  # 72 move types
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Generate queries and keys for each square
        queries = self.query_proj(x)  # [batch, 64, 256]
        keys = self.key_proj(x)  # [batch, 64, 256]
        
        # Compute attention scores between all square pairs
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(256)
        # scores shape: [batch, 64, 64]
        
        # Get context features
        context = self.context_proj(x)  # [batch, 64, 128]
        
        # Generate policy logits for each square and move type
        policy_per_square = self.policy_proj(context)  # [batch, 64, 72]
        
        # Flatten to standard policy output format
        policy_flat = policy_per_square.reshape(batch_size, 64 * 72)
        
        return policy_flat


class ValueHead(nn.Module):
    """
    Value head for position evaluation.
    """
    
    def __init__(self, d_model):
        super().__init__()
        
        self.global_pool = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.value_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Global average pooling
        x = x.mean(dim=1)  # [batch, d_model]
        
        # Project to value
        x = self.global_pool(x)
        value = self.value_proj(x)
        
        return value


class Luna(nn.Module):
    """
    Transformer-based neural network for chess.
    Target: ~240M parameters for stronger play.
    """
    
    def __init__(
        self,
        num_layers=15,
        d_model=1024,
        num_heads=32,
        d_ff=4096,
        dropout=0.1,
        policy_weight=1.0,
        input_planes=16,
        use_rope=False,
        use_alibi=False,
        use_cls_token=True,
        entropy_coef=0.0,
        use_gradient_checkpointing=False,
    ):
        super().__init__()

        self.policy_weight = policy_weight
        self.d_model = d_model
        self.entropy_coef = entropy_coef
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Input projection from N-plane encoding to tokens
        self.input_projection = nn.Conv2d(input_planes, d_model, kernel_size=1)
        
        # Positional encoding
        self.positional_encoding = ChessPositionalEncoding(d_model)
        
        # Smolgen-inspired projection
        self.smolgen = SmolgenInspiredProjection(d_model, num_heads)
        
        # Optional CLS token for global pooling
        self.use_cls = use_cls_token
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, use_rope=use_rope, use_alibi=use_alibi)
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.value_head = ValueHead(d_model)
        self.policy_head = PolicyHead(d_model)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, valueTarget=None, policyTarget=None, policyMask=None):
        batch_size = x.shape[0]
        
        # Project input planes to tokens
        x = self.input_projection(x)  # [batch, d_model, 8, 8]
        
        # Reshape to sequence of tokens
        x = x.view(batch_size, self.d_model, 64).transpose(1, 2)  # [batch, 64, d_model]
        # Prepend CLS token if enabled
        if self.use_cls:
            cls = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls, x], dim=1)  # [batch, 65, d_model]
        
        # Add positional encoding (board tokens only); keep CLS as learnable only
        if self.use_cls:
            cls_tok, board_tok = x[:, :1, :], x[:, 1:, :]
            board_tok = self.positional_encoding(board_tok)
            x = torch.cat([cls_tok, board_tok], dim=1)
        else:
            x = self.positional_encoding(x)
        
        # Add smolgen-inspired features (apply to board tokens only if CLS is present)
        if self.use_cls:
            cls_tok, board_tok = x[:, :1, :], x[:, 1:, :]
            smolgen_features = self.smolgen(board_tok)
            board_tok = board_tok + 0.1 * smolgen_features
            x = torch.cat([cls_tok, board_tok], dim=1)
        else:
            smolgen_features = self.smolgen(x)
            x = x + 0.1 * smolgen_features  # Small contribution initially
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to reduce activation memory
                def _block_fn(inp):
                    return block(inp)
                x = torch.utils.checkpoint.checkpoint(_block_fn, x)
            else:
                x = block(x)
        
        # Get value and policy outputs
        if self.use_cls:
            value = self.value_head(x[:, :1, :])  # pool from CLS
            policy = self.policy_head(x[:, 1:, :])  # policy from board tokens
        else:
            value = self.value_head(x)
            policy = self.policy_head(x)
        
        if valueTarget is not None and policyTarget is not None:
            # Calculate losses
            valueLoss = self.mse_loss(value, valueTarget)

            # Optionally apply legal move mask to logits (improves sample efficiency)
            masked_policy = policy
            if policyMask is not None:
                pm = policyMask.view(batch_size, -1).float()
                # avoid -inf in autograd; use large negative where illegal
                large_neg = -1e9
                masked_policy = masked_policy + (pm + 1e-8).log()  # log(0) -> -inf, approximated
            
            # Handle different policy target formats
            if policyTarget.dim() == 1 or (policyTarget.dim() == 2 and policyTarget.shape[1] == 1):
                # Supervised learning: policy target is move index
                policyTarget = policyTarget.view(batch_size).long()
                policyLoss = self.cross_entropy_loss(masked_policy, policyTarget)
            else:
                # RL: policy target is probability distribution
                if policyMask is not None:
                    # Renormalize targets over legal moves only
                    pm = policyMask.view(batch_size, -1).float()
                    policyTarget = policyTarget * pm
                # Normalize target distribution
                policyTarget = policyTarget + 1e-8
                policyTarget = policyTarget / policyTarget.sum(dim=1, keepdim=True)
                log_probs = F.log_softmax(masked_policy, dim=1)
                policyLoss = -(policyTarget * log_probs).sum(dim=1).mean()
            
            # Optional entropy regularization on masked policy logits
            ent = 0.0
            if self.entropy_coef > 0.0:
                log_probs = F.log_softmax(masked_policy, dim=1)
                probs = log_probs.exp()
                ent = -(probs * log_probs).sum(dim=1).mean()

            totalLoss = valueLoss + self.policy_weight * policyLoss - self.entropy_coef * ent
            
            return totalLoss, valueLoss, policyLoss
        
        else:
            # Inference mode
            if policyMask is not None:
                policyMask = policyMask.view(batch_size, -1)
                
                # Apply mask and numerically stable softmax over legal moves only
                shifted = policy - policy.max(dim=1, keepdim=True).values
                policy_exp = torch.exp(shifted) * policyMask.float()
                sums = policy_exp.sum(dim=1, keepdim=True)
                # Avoid divide-by-zero when mask is empty by falling back to ones
                sums = torch.where(sums > 0, sums, torch.ones_like(sums))
                policy_softmax = policy_exp / sums
            else:
                policy_softmax = F.softmax(policy, dim=1)
            
            return value, policy_softmax


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = Luna()
    print(f"Luna initialized with {count_parameters(model):,} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(2, 16, 8, 8)
    dummy_value_target = torch.randn(2, 1)
    dummy_policy_target = torch.randint(0, 4608, (2,))
    dummy_mask = torch.randint(0, 2, (2, 72, 8, 8))
    
    # Training mode
    model.train()
    total_loss, value_loss, policy_loss = model(
        dummy_input, 
        valueTarget=dummy_value_target,
        policyTarget=dummy_policy_target,
        policyMask=dummy_mask
    )
    print(f"Training - Total loss: {total_loss:.4f}, Value loss: {value_loss:.4f}, Policy loss: {policy_loss:.4f}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        value, policy = model(dummy_input, policyMask=dummy_mask)
        print(f"Inference - Value shape: {value.shape}, Policy shape: {policy.shape}")