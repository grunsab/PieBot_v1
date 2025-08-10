import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. Core Building Blocks for an Efficient CNN
# These modules are designed to be lightweight and fast on CPUs.
# ==============================================================================

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation module for channel-wise attention.
    This lightweight module helps the network focus on the most informative
    features with minimal computational overhead.
    """
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        # The number of channels in the bottleneck layer.
        squeeze_channels = max(1, channels // reduction_ratio)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, squeeze_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze_channels, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution. This is the key to an efficient
    CPU-based model. It splits a standard convolution into two much smaller
    operations (a depthwise and a pointwise convolution), dramatically
    reducing parameters and computation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size, 
            stride, 
            padding, 
            groups=in_channels, # This makes it a depthwise convolution
            bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SEResidualBlock(nn.Module):
    """
    An efficient residual block that combines Squeeze-Excitation with
    Depthwise Separable Convolutions. This is the core of our lightweight network.
    """
    def __init__(self, num_filters, use_se=True, dropout_rate=0.0):
        super().__init__()
        
        self.conv1 = DepthwiseSeparableConv2d(num_filters, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        self.conv2 = DepthwiseSeparableConv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.se = SqueezeExcitation(num_filters) if use_se else None
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.dropout:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.se:
            out = self.se(out)
            
        out += residual
        out = self.relu(out)
        
        return out

# ==============================================================================
# 2. Optimized Output Heads
# These are simplified versions of the modern WDL and Convolutional heads.
# ==============================================================================

class WDLValueHead(nn.Module):
    """
    A Win-Draw-Loss (WDL) value head. It provides a much richer evaluation
    signal to the search algorithm than a single scalar value, which is
    critical for strong play, even in a small network.
    """
    def __init__(self, input_channels, hidden_dim=128):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 16, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(16 * 64, hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 3) # Output logits for Win, Draw, Loss
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x

class ConvolutionalPolicyHead(nn.Module):
    """
    A truly lightweight convolutional policy head for PieNano.
    Directly outputs 73 planes × 8×8 = 4672 logits without a large FC layer.
    The MCTS/UCI engine will handle mapping to the 4608 legal moves.
    """
    def __init__(self, input_channels, num_moves=4608):
        super().__init__()
        # The number of possible move types from any given square.
        # We output 72 planes (not 73) to match the exact 4608 moves (72*64=4608)
        num_move_planes = 72
        self.conv = nn.Conv2d(input_channels, num_move_planes, kernel_size=1, bias=True)
        self.num_moves = num_moves
        
    def forward(self, x):
        x = self.conv(x)
        # Flatten to (batch, 72*64) = (batch, 4608)
        x = x.view(x.size(0), -1)
        return x

# ==============================================================================
# 3. Main Model: PieBotNano
# This class assembles the efficient components into the final network.
# ==============================================================================

class PieNano(nn.Module):
    """
    A lightweight, CPU-efficient neural network for a chess bot.

    Key Architectural Features:
    - Depthwise Separable Convolutions for speed.
    - Squeeze-and-Excitation blocks for channel attention.
    - A small residual tower (8 blocks, 128 filters) for a good speed/strength balance.
    - Modern WDL Value and Convolutional Policy heads for high-quality search guidance.
    """
    def __init__(self, 
                 num_blocks=8, 
                 num_filters=128, 
                 num_input_planes=16,
                 num_moves=4608,
                 use_se=True,
                 dropout_rate=0.0,
                 policy_weight=1.0):
        super().__init__()
        
        self.policy_weight = policy_weight
        
        # Initial convolution block (standard conv) to create the feature space.
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_input_planes, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        # The main body of the network is a stack of our efficient residual blocks.
        self.residual_tower = nn.Sequential(
            *[SEResidualBlock(num_filters, use_se=use_se, dropout_rate=dropout_rate) for _ in range(num_blocks)])
               
        # The modern, efficient output heads.
        self.value_head = WDLValueHead(num_filters)
        self.policy_head = ConvolutionalPolicyHead(num_filters, num_moves=num_moves)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initializes weights using He initialization, common for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x, value_targets=None, policy_targets=None, policyMask=None):
        """
        Performs the forward pass of the network.
        
        Args:
            x: Input board positions
            value_targets: Target values for training (optional)
            policy_targets: Target policies for training (optional)
            policyMask: Legal move mask (optional, not used by PieNano but accepted for compatibility)
            
        Returns:
            During training: (loss, value_loss, policy_loss)
            During inference: (policy_logits, value_logits)
        """
        # 1. Initial feature extraction
        x = self.conv_block(x)
        
        # 2. Pass through the residual tower
        x = self.residual_tower(x)
        
        # 3. Compute value and policy logits
        value_logits = self.value_head(x)
        policy_logits = self.policy_head(x)
        
        # If targets are provided, compute losses
        if value_targets is not None and policy_targets is not None:
            # Value loss (WDL uses CrossEntropy)
            # Convert value targets to WDL format if needed
            if value_targets.dim() == 1 or (value_targets.dim() == 2 and value_targets.size(1) == 1):
                # Convert scalar values to WDL (win/draw/loss)
                # Assumes value_targets are in [-1, 1] range
                wdl_targets = torch.zeros(value_targets.size(0), 3, device=value_targets.device)
                value_targets = value_targets.view(-1)
                
                # Simple conversion: map [-1, 1] to WDL probabilities
                # Win probability: (value + 1) / 2
                # Loss probability: (1 - value) / 2  
                # Draw probability: peaked around 0
                draw_prob = torch.exp(-4 * value_targets**2)  # Gaussian around 0
                win_prob = torch.clamp((value_targets + 1) / 2 * (1 - draw_prob), 0, 1)
                loss_prob = torch.clamp((1 - value_targets) / 2 * (1 - draw_prob), 0, 1)
                
                wdl_targets[:, 0] = win_prob
                wdl_targets[:, 1] = draw_prob
                wdl_targets[:, 2] = loss_prob
                
                # Normalize to ensure sum = 1
                wdl_targets = wdl_targets / wdl_targets.sum(dim=1, keepdim=True)
                value_targets = wdl_targets
            
            # WDL loss (cross entropy with soft targets)
            value_loss = F.cross_entropy(value_logits, value_targets.argmax(dim=1))
            
            # Policy loss
            if policy_targets.dim() == 1:
                # Targets are indices
                policy_loss = F.cross_entropy(policy_logits, policy_targets)
            else:
                # Targets are distributions (soft targets)
                log_probs = F.log_softmax(policy_logits, dim=1)
                policy_loss = -(policy_targets * log_probs).sum() / policy_targets.size(0)
            
            # Combined loss
            total_loss = value_loss + self.policy_weight * policy_loss
            
            return total_loss, value_loss, policy_loss
        else:
            # Inference mode - return value first (as expected by MCTS)
            # Convert WDL logits to scalar value for MCTS
            # Apply softmax to get probabilities
            wdl_probs = F.softmax(value_logits, dim=1)
            # Convert to scalar: win_prob - loss_prob (draw_prob is neutral)
            value_scalar = wdl_probs[:, 0:1] - wdl_probs[:, 2:3]  # Keep dimensions for compatibility
            return value_scalar, policy_logits

