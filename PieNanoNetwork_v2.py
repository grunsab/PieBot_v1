"""
PieNano V2: A balanced lightweight architecture with better policy head.
Targets ~2-3M parameters for good performance on VPS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the efficient building blocks from original PieNano
from PieNanoNetwork import SqueezeExcitation, DepthwiseSeparableConv2d, SEResidualBlock, WDLValueHead

class ImprovedPolicyHead(nn.Module):
    """
    Balanced policy head with small FC layer for better move correlation learning.
    Uses dimensionality reduction to keep parameters manageable.
    """
    def __init__(self, input_channels, num_moves=4608, hidden_dim=256):
        super().__init__()
        # First reduce channels significantly
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Small FC layers for move correlation
        self.fc1 = nn.Linear(32 * 64, hidden_dim)  # 2048 -> 256
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, num_moves)  # 256 -> 4608
        
    def forward(self, x):
        # Reduce dimensions with conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Flatten and pass through FC layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PieNanoV2(nn.Module):
    """
    PieNano V2: Balanced architecture for VPS deployment.
    ~2-3M parameters with better policy head.
    """
    def __init__(self, 
                 num_blocks=8, 
                 num_filters=128, 
                 num_input_planes=16,
                 num_moves=4608,
                 use_se=True,
                 dropout_rate=0.0,
                 policy_weight=1.0,
                 policy_hidden_dim=256):
        super().__init__()
        
        self.policy_weight = policy_weight
        
        # Initial convolution block
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_input_planes, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        # Residual tower
        self.residual_tower = nn.Sequential(
            *[SEResidualBlock(num_filters, use_se=use_se, dropout_rate=dropout_rate) 
              for _ in range(num_blocks)]
        )
        
        # Output heads
        self.value_head = WDLValueHead(num_filters, hidden_dim=128)
        self.policy_head = ImprovedPolicyHead(num_filters, num_moves, hidden_dim=policy_hidden_dim)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
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
    
    def forward(self, x, value_targets=None, policy_targets=None):
        """Forward pass with optional loss computation."""
        # Feature extraction
        x = self.conv_block(x)
        x = self.residual_tower(x)
        
        # Compute heads
        value_logits = self.value_head(x)
        policy_logits = self.policy_head(x)
        
        # Return based on mode
        if value_targets is not None and policy_targets is not None:
            # Training mode - compute losses
            return self._compute_loss(value_logits, policy_logits, value_targets, policy_targets)
        else:
            # Inference mode
            return policy_logits, value_logits
    
    def _compute_loss(self, value_logits, policy_logits, value_targets, policy_targets):
        """Compute training losses."""
        # Value loss (WDL)
        if value_targets.dim() == 1 or (value_targets.dim() == 2 and value_targets.size(1) == 1):
            # Convert scalar to WDL
            wdl_targets = torch.zeros(value_targets.size(0), 3, device=value_targets.device)
            value_targets = value_targets.view(-1)
            
            draw_prob = torch.exp(-4 * value_targets**2)
            win_prob = torch.clamp((value_targets + 1) / 2 * (1 - draw_prob), 0, 1)
            loss_prob = torch.clamp((1 - value_targets) / 2 * (1 - draw_prob), 0, 1)
            
            wdl_targets[:, 0] = win_prob
            wdl_targets[:, 1] = draw_prob
            wdl_targets[:, 2] = loss_prob
            
            wdl_targets = wdl_targets / wdl_targets.sum(dim=1, keepdim=True)
            value_targets = wdl_targets
        
        value_loss = F.cross_entropy(value_logits, value_targets.argmax(dim=1))
        
        # Policy loss
        if policy_targets.dim() == 1:
            policy_loss = F.cross_entropy(policy_logits, policy_targets)
        else:
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(policy_targets * log_probs).sum() / policy_targets.size(0)
        
        total_loss = value_loss + self.policy_weight * policy_loss
        
        return total_loss, value_loss, policy_loss


# Test the models
if __name__ == "__main__":
    print("Comparing PieNano architectures:\n")
    
    # Original PieNano (ultra-light)
    from PieNanoNetwork import PieNano
    model_v1 = PieNano(num_blocks=8, num_filters=128)
    params_v1 = sum(p.numel() for p in model_v1.parameters())
    
    # PieNano V2 (balanced)
    model_v2 = PieNanoV2(num_blocks=8, num_filters=128, policy_hidden_dim=256)
    params_v2 = sum(p.numel() for p in model_v2.parameters())
    
    print(f"PieNano V1 (Ultra-light):")
    print(f"  Parameters: {params_v1:,}")
    print(f"  Size (FP32): {params_v1 * 4 / 1024 / 1024:.2f} MB")
    print(f"  Size (INT8): {params_v1 / 1024 / 1024:.2f} MB")
    
    print(f"\nPieNano V2 (Balanced):")
    print(f"  Parameters: {params_v2:,}")
    print(f"  Size (FP32): {params_v2 * 4 / 1024 / 1024:.2f} MB")
    print(f"  Size (INT8): {params_v2 / 1024 / 1024:.2f} MB")
    
    # Break down V2 parameters
    policy_params = sum(p.numel() for n, p in model_v2.named_parameters() if 'policy_head' in n)
    print(f"\nPieNano V2 Policy head: {policy_params:,} parameters")
    print(f"  (Manageable but can learn move correlations)")
    
    print(f"\nRecommendation:")
    print(f"  V1: Use if RAM is critical (<5GB) or max speed needed")
    print(f"  V2: Use for better chess strength while still lightweight")