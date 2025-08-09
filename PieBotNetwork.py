import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation module for channel-wise attention.
    Helps the network focus on the most informative features.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution for efficient computation.
    Reduces parameters by ~8-9x compared to standard convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ConvBlock(nn.Module):
    """
    Enhanced convolutional block with optional depthwise separable convolution.
    """
    def __init__(self, input_channels, num_filters, use_depthwise=False):
        super().__init__()
        if use_depthwise:
            self.conv = DepthwiseSeparableConv2d(input_channels, num_filters, 3, padding=1)
        else:
            self.conv = nn.Conv2d(input_channels, num_filters, 3, padding=1)
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SEResidualBlock(nn.Module):
    """
    Enhanced residual block with Squeeze-Excitation and optional depthwise separable convolutions.
    """
    def __init__(self, num_filters, use_se=True, use_depthwise=True, dropout_rate=0.0):
        super().__init__()
        
        if use_depthwise:
            self.conv1 = DepthwiseSeparableConv2d(num_filters, num_filters, 3, padding=1)
            self.conv2 = DepthwiseSeparableConv2d(num_filters, num_filters, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
            self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
            
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        
        self.se = SqueezeExcitation(num_filters) if use_se else None
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        if self.dropout:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.se:
            x = self.se(x)
            
        x += residual
        x = self.relu(x)
        
        return x

class ChessPositionalEncoding(nn.Module):
    """
    Chess-specific positional encoding inspired by smolgen from Leela Chess Zero.
    Provides spatial awareness for the 8x8 board.
    """
    def __init__(self, num_filters):
        super().__init__()
        self.num_filters = num_filters
        
        # Learnable positional embeddings for each square
        self.pos_embedding = nn.Parameter(torch.randn(1, num_filters, 8, 8))
        
        # Additional learnable embeddings for files and ranks
        self.file_embedding = nn.Parameter(torch.randn(1, num_filters // 4, 1, 8))
        self.rank_embedding = nn.Parameter(torch.randn(1, num_filters // 4, 8, 1))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Expand positional embeddings to batch size
        pos_emb = self.pos_embedding.expand(batch_size, -1, -1, -1)
        
        # Expand and broadcast file/rank embeddings
        file_emb = self.file_embedding.expand(batch_size, -1, 8, -1)
        rank_emb = self.rank_embedding.expand(batch_size, -1, -1, 8)
        
        # Combine file and rank embeddings and repeat to match num_filters
        file_rank_emb = torch.cat([
            file_emb,
            rank_emb
        ], dim=1)
        
        # Repeat the file_rank_emb to match num_filters
        # file_rank_emb has num_filters // 2 channels, so repeat twice
        file_rank_emb = file_rank_emb.repeat(1, 2, 1, 1)
        
        # Add positional information to input
        return x + pos_emb + file_rank_emb

class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block adapted for chess positions.
    Uses multi-head self-attention with chess-specific positional encoding.
    """
    def __init__(self, num_filters, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_filters = num_filters
        self.num_heads = num_heads
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(num_filters)
        self.ln2 = nn.LayerNorm(num_filters)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            num_filters, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network with GELU activation
        self.ffn = nn.Sequential(
            nn.Linear(num_filters, num_filters * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters * 2, num_filters),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        b, c, h, w = x.shape
        
        # Reshape to sequence format for transformer
        x_seq = x.view(b, c, h * w).permute(0, 2, 1)  # (batch, 64, channels)
        
        # Apply layer norm and self-attention
        x_norm = self.ln1(x_seq)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_out
        
        # Apply layer norm and feed-forward network
        x_norm = self.ln2(x_seq)
        x_seq = x_seq + self.ffn(x_norm)
        
        # Reshape back to spatial format
        x = x_seq.permute(0, 2, 1).view(b, c, h, w)
        
        return x

class ImprovedValueHead(nn.Module):
    """
    Enhanced value head with additional capacity and regularization.
    """
    def __init__(self, input_channels, hidden_dim=512):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 32, 1)
        self.bn = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(32 * 64, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.fc3(x)
        x = self.tanh(x)
        return x

class ImprovedPolicyHead(nn.Module):
    """
    Enhanced policy head with better feature extraction.
    """
    def __init__(self, input_channels, num_moves=4608):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 32, 1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(32 * 64, num_moves)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class PieBotNet(nn.Module):
    """
    Enhanced AlphaZero-style neural network with modern improvements:
    - Squeeze-Excitation attention mechanisms
    - Depthwise separable convolutions for efficiency
    - Transformer encoder blocks for global context
    - Chess-specific positional encoding
    - Improved regularization and normalization
    """
    def __init__(self, 
                 num_blocks=24, 
                 num_filters=256, 
                 num_transformer_blocks=2,
                 num_input_planes=112,
                 use_se=True,
                 use_depthwise=True,
                 dropout_rate=0.1,
                 policy_weight=1.0):
        super().__init__()
        
        self.policy_weight = policy_weight
        
        # Initial convolution block
        self.conv_block = ConvBlock(num_input_planes, num_filters, use_depthwise=False)
        
        # Residual tower with SE blocks
        residual_blocks = []
        for i in range(num_blocks):
            # Use more depthwise convolutions in later blocks for efficiency
            use_dw = use_depthwise and (i >= num_blocks // 4)
            residual_blocks.append(
                SEResidualBlock(num_filters, use_se=use_se, 
                              use_depthwise=use_dw, dropout_rate=dropout_rate)
            )
        self.residual_blocks = nn.ModuleList(residual_blocks)
        
        # Chess-specific positional encoding
        self.positional_encoding = ChessPositionalEncoding(num_filters)
        
        # Transformer encoder blocks for global context
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(num_filters, num_heads=8, dropout=dropout_rate)
            for _ in range(num_transformer_blocks)
        ])
        
        # Output heads
        self.value_head = ImprovedValueHead(num_filters)
        self.policy_head = ImprovedPolicyHead(num_filters)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Initialize weights using He initialization for ReLU networks.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x, value_target=None, policy_target=None, policy_mask=None):
        # Initial convolution
        x = self.conv_block(x)
        
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
            
        # Add positional encoding before transformer blocks
        x = self.positional_encoding(x)
        
        # Apply transformer blocks for global context
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Compute value and policy
        value = self.value_head(x)
        policy = self.policy_head(x)
        
        if self.training:
            # Calculate losses during training
            value_loss = self.mse_loss(value, value_target)
            
            # Handle different policy target formats
            if policy_target.dim() == 1 or (policy_target.dim() == 2 and policy_target.shape[1] == 1):
                # Supervised learning: policy target is move index
                policy_target = policy_target.view(policy_target.shape[0])
                policy_loss = self.cross_entropy_loss(policy, policy_target)
            else:
                # RL: policy target is probability distribution
                log_probs = F.log_softmax(policy, dim=1)
                policy_target = policy_target + 1e-8
                policy_target = policy_target / policy_target.sum(dim=1, keepdim=True)
                policy_loss = -(policy_target * log_probs).sum(dim=1).mean()
            
            total_loss = value_loss + self.policy_weight * policy_loss
            return total_loss, value_loss, policy_loss
        else:
            # Inference mode
            if policy_mask is not None:
                policy_mask = policy_mask.view(policy_mask.shape[0], -1)
                policy_exp = torch.exp(policy)
                policy_exp *= policy_mask.type(torch.float32)
                policy_exp_sum = torch.sum(policy_exp, dim=1, keepdim=True)
                policy_softmax = policy_exp / (policy_exp_sum + 1e-8)
            else:
                policy_softmax = F.softmax(policy, dim=1)
                
            return value, policy_softmax

class PieBotNetConfig:
    """
    Configuration class for PieBotNet hyperparameters.
    """
    def __init__(self):
        self.num_blocks = 24
        self.num_filters = 256
        self.num_transformer_blocks = 2
        self.num_input_planes = 112
        self.use_se = True
        self.use_depthwise = True
        self.dropout_rate = 0.1
        self.policy_weight = 1.0
        
        # Training configuration
        self.use_mixed_precision = True
        self.gradient_accumulation_steps = 4
        self.warmup_steps = 1000
        self.cosine_annealing = True
        self.label_smoothing = 0.1