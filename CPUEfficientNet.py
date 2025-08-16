import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import fuse_modules
import os

# -----------------------------
# Building Blocks
# -----------------------------

class Stem(nn.Module):
    """
    3x3 stem: 16 input planes -> F channels.
    ReLU6 is quantization-friendly (clamps activation range).
    """
    def __init__(self, out_ch: int):
        super().__init__()
        # bias=False because BatchNorm follows immediately, required for Conv-BN fusion
        self.conv = nn.Conv2d(16, out_ch, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU6(inplace=True)

    def forward(self, x):
        # Explicit module calls are required for fusion tracking
        x = self.conv(x); x = self.bn(x); x = self.act(x)
        return x

    def fuse_model(self):
        fuse_modules(self, ['conv', 'bn', 'act'], inplace=True)


class SqueezeExcitation(nn.Module):
    """
    SE block implemented with 1x1 convs (good for quantization).
    """
    def __init__(self, input_channels: int, squeeze_channels: int):
        super().__init__()
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1, bias=True)
        self.act = nn.ReLU6(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # Squeeze (Global Average Pooling)
        scale = F.adaptive_avg_pool2d(x, 1)
        # Excitation
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.sig(scale)
        # Scale (Multiplication is handled correctly by quantization)
        return x * scale

    def fuse_model(self):
        # Fuse conv + relu6 in the "excitation" part (fc1 + act)
        # Conv+ReLU fusion is supported even without BN.
        fuse_modules(self, ['fc1', 'act'], inplace=True)


class CPUEfficientBlock(nn.Module):
    """
    Hybrid DW/MBConv residual block, optimized for CPU and quantization.

    If expansion_factor == 1: Standard Depthwise Separable (MobileNetV1 style).
    If expansion_factor > 1: Inverted Residual Bottleneck (MBConv/MobileNetV2 style).
    
    Features: Linear bottlenecks, optional SE, and explicit handling of residual addition for quantization.
    """
    def __init__(self,
                 channels: int,
                 expansion_factor: int = 1,
                 se_ratio: float = 0.0):
        super().__init__()
        self.channels = channels
        self.e = int(expansion_factor)
        self.use_se = se_ratio > 0.0

        hidden = channels * self.e

        # 1. Optional expansion (MBConv)
        if self.e > 1:
            self.expand_conv = nn.Conv2d(channels, hidden, 1, bias=False)
            self.bn0         = nn.BatchNorm2d(hidden)
            self.act0        = nn.ReLU6(inplace=True)
        else:
            self.expand_conv = None

        # Determine the input/output channels for the depthwise stage
        input_dim = hidden if self.e > 1 else channels

        # 2. Depthwise (3x3)
        self.dw = nn.Conv2d(input_dim, input_dim, 3, padding=1, groups=input_dim, bias=False)
        self.bn1  = nn.BatchNorm2d(input_dim)
        self.act1 = nn.ReLU6(inplace=True)

        # 3. Squeeze-and-Excitation
        if self.use_se:
            # SE applied to the expanded dimension (EfficientNet style).
            se_ch = max(1, int(input_dim * se_ratio))
            self.se = SqueezeExcitation(input_dim, se_ch)
        else:
            self.se = None

        # 4. Projection / Pointwise (Linear Bottleneck)
        self.pw = nn.Conv2d(input_dim, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # Linear bottleneck: no activation here

        # CRITICAL: For quantization, residual additions must use FloatFunctional
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x

        out = x
        if self.expand_conv is not None:
            out = self.expand_conv(out); out = self.bn0(out); out = self.act0(out)

        out = self.dw(out); out = self.bn1(out); out = self.act1(out)

        if self.se is not None:
            out = self.se(out)

        out = self.pw(out); out = self.bn2(out)

        # Use FloatFunctional for the residual addition
        out = self.skip_add.add(out, residual)
        return out

    def fuse_model(self):
        # Batch the fusion calls where possible
        modules_to_fuse = []
        if self.expand_conv is not None:
             modules_to_fuse.append(['expand_conv', 'bn0', 'act0'])
             
        modules_to_fuse.append(['dw', 'bn1', 'act1'])
        # Fuse the linear projection (Conv-BN only)
        modules_to_fuse.append(['pw', 'bn2'])
        
        if modules_to_fuse:
            fuse_modules(self, modules_to_fuse, inplace=True)
        
        if self.se is not None:
            self.se.fuse_model()


class ValueHead(nn.Module):
    """
    1x1 -> BN -> ReLU6 -> FC(64->256) -> ReLU6 -> FC(256->1) -> Tanh
    """
    def __init__(self, input_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 1, 1, bias=False)
        self.bn   = nn.BatchNorm2d(1)
        self.act  = nn.ReLU6(inplace=True)
        self.fc1  = nn.Linear(64, 256, bias=True)
        self.act2 = nn.ReLU6(inplace=True)
        self.fc2  = nn.Linear(256, 1, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = self.act(x)
        # 8x8 board -> 64 features
        x = x.view(x.shape[0], 64)
        x = self.fc1(x); x = self.act2(x)
        x = self.fc2(x)
        return self.tanh(x)

    def fuse_model(self):
        fuse_modules(self, ['conv', 'bn', 'act'], inplace=True)
        # Fusion of Linear-ReLU is also supported in modern PyTorch quantization
        fuse_modules(self, ['fc1', 'act2'], inplace=True)


class PolicyHead(nn.Module):
    """
    1x1 -> BN -> ReLU6 -> flatten(128) -> FC(128->4608)
    """
    def __init__(self, input_channels: int):
        super().__init__()
        # 2 channels for the policy head convolution (as in AlphaZero)
        self.conv = nn.Conv2d(input_channels, 2, 1, bias=False)
        self.bn   = nn.BatchNorm2d(2)
        self.act  = nn.ReLU6(inplace=True)
        # 2 channels * 8x8 board = 128 features
        self.fc   = nn.Linear(128, 4608, bias=True)

    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = self.act(x)
        x = x.reshape(x.shape[0], 128)
        return self.fc(x)

    def fuse_model(self):
        fuse_modules(self, ['conv', 'bn', 'act'], inplace=True)


# -----------------------------
# Main Network
# -----------------------------

class CPUEfficientNet(nn.Module):
    """
    A CPU-first AlphaZero-style network designed for efficiency and INT8 quantization.

    Parameters:
      num_blocks:        Total number of residual blocks.
      num_filters:       Base number of channels (C).
      expansion_factor:  Factor (e) to expand channels in MBConv blocks (if >1).
      se_ratio:          Squeeze ratio for SE blocks (0 to disable; e.g., 0.125 or 0.25).
      mbconv_every:      If >=2, apply MBConv (e>1) every kth block, use DW (e=1) otherwise.
                         If 0, use the same expansion_factor for all blocks.
    """
    def __init__(self,
                 num_blocks: int,
                 num_filters: int,
                 policy_weight: float = 1.0,
                 expansion_factor: int = 4, # Default to a standard MBConv expansion
                 se_ratio: float = 0.125,
                 mbconv_every: int = 0):    # Default to uniform architecture
        super().__init__()
        self.policy_weight = policy_weight

        # Input planes fixed at 16
        self.stem = Stem(num_filters)

        # Build the residual tower with flexible configuration
        blocks = []
        for i in range(num_blocks):
            # Determine the expansion factor for the current block
            # This logic ensures that if hybrid architecture is used (mbconv_every >= 2), 
            # scheduled blocks are strong (e>=2) and others are fast (e=1).
            if mbconv_every and mbconv_every >= 2 and ((i + 1) % mbconv_every == 0):
                # Scheduled block: Use the strong expansion factor (enforce at least 2)
                e = max(2, expansion_factor)
            else:
                # Non-scheduled block: Use e=1 if hybrid (mbconv_every!=0), otherwise use the global factor
                e = 1 if mbconv_every else max(1, expansion_factor)
                
            blocks.append(CPUEfficientBlock(num_filters, expansion_factor=e, se_ratio=se_ratio))
        self.blocks = nn.ModuleList(blocks)

        self.valueHead  = ValueHead(num_filters)
        self.policyHead = PolicyHead(num_filters)

        # Losses
        self.mseLoss = nn.MSELoss()
        self.crossEntropyLoss = nn.CrossEntropyLoss()

        # Quantization stubs (Define FP32 <-> INT8 boundaries)
        self.quant = torch.ao.quantization.QuantStub()
        # Use separate stubs for value/policy as they have different output distributions/ranges
        self.dequant_value = torch.ao.quantization.DeQuantStub()
        self.dequant_policy = torch.ao.quantization.DeQuantStub()

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x, valueTarget=None, policyTarget=None, policyMask=None):
        # Quantize input (converts FP32 to INT8 if the model is quantized)
        x = self.quant(x)

        x = self.stem(x)
        for b in self.blocks:
            x = b(x)

        value = self.valueHead(x)
        policy_logits = self.policyHead(x)

        # Dequantize outputs (converts INT8 to FP32)
        value = self.dequant_value(value)
        policy_logits = self.dequant_policy(policy_logits)

        if self.training:
            # Training logic (Handles both SL and RL/Distillation)
            valueLoss = self.mseLoss(value, valueTarget)

            if policyTarget.dim() == 1 or (policyTarget.dim() == 2 and policyTarget.shape[1] == 1):
                # Supervised learning (index target)
                policyTarget = policyTarget.view(policyTarget.shape[0])
                policyLoss = self.crossEntropyLoss(policy_logits, policyTarget)
            else:
                # RL/Distillation (distributional target) - Cross-entropy with soft targets
                log_probs = F.log_softmax(policy_logits, dim=1)
                # Add epsilon for stability and normalize
                p = policyTarget + 1e-8
                p = p / p.sum(dim=1, keepdim=True)
                policyLoss = -(p * log_probs).sum(dim=1).mean()

            total = valueLoss + self.policy_weight * policyLoss
            return total, valueLoss, policyLoss

        # Inference: Logit masking pre-softmax (numerically stable and fast)
        if policyMask is not None:
            policyMask = policyMask.view(policy_logits.shape[0], -1)
            # Set logits of illegal moves to negative infinity
            policy_logits = policy_logits.masked_fill(policyMask == 0, float('-inf'))

        policy_probs = F.softmax(policy_logits, dim=1)
        return value, policy_probs

    # -------------------------
    # Perf / Deployment helpers
    # -------------------------
    def fuse_model(self):
        """
        Fuses Conv+BN(+ReLU6) and Linear(+ReLU6) modules for inference optimization and quantization.
        Call this before quantization (PTQ or QAT) and ensure the model is in eval mode.
        """
        # Must be in eval mode for BN fusion to work correctly
        self.eval() 
        self.stem.fuse_model()
        for b in self.blocks:
            b.fuse_model()
        self.valueHead.fuse_model()
        self.policyHead.fuse_model()

    def to_channels_last(self):
        """
        Changes memory layout to NHWC (channels_last), often faster on ARM CPUs (QNNPACK).
        """
        self.to(memory_format=torch.channels_last)
        
        # Ensure weights and biases are contiguous in the new format after the change
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
                if m.bias is not None:
                    # Bias is 1D, standard contiguous is sufficient
                    m.bias.data = m.bias.data.contiguous()
            elif isinstance(m, nn.Linear):
                 m.weight.data = m.weight.data.contiguous()
                 if m.bias is not None:
                    m.bias.data = m.bias.data.contiguous()

    @staticmethod
    def set_cpu_threads(n=None):
        """Sets the number of threads used for intra-op parallelism."""
        if n is None:
            n = os.cpu_count() or 4
        torch.set_num_threads(n)