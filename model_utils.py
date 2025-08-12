"""
Shared utilities for model detection and loading across the codebase.
Handles AlphaZeroNet, PieBotNet, PieNano, PieNanoV2, and TitanMini models.
Supports both regular and quantized model formats.
"""

import torch
import torch.nn as nn
import AlphaZeroNetwork
import PieBotNetwork
import PieNanoNetwork
import PieNanoNetwork_v2
import TitanMiniNetwork


def detect_model_type(weights, model_path=None):
    """
    Detect the type of chess model from weights or filename.
    
    Args:
        weights: Loaded model weights (dict or model object)
        model_path: Optional path to model file for filename-based detection
    
    Returns:
        str: Model type identifier
    """
    # Check if it's a quantized model by filename
    if model_path and ('quantized' in model_path.lower() or 'quant' in model_path.lower() or '_q' in model_path.lower()):
        # Try to determine the base model type from filename
        if 'titan' in model_path.lower() or 'titanmini' in model_path.lower():
            return 'TitanMini_Quantized'
        elif 'pienano' in model_path.lower() or 'pie_nano' in model_path.lower():
            return 'PieNano_Quantized'
        elif 'alphazero' in model_path.lower() or 'alpha_zero' in model_path.lower():
            return 'AlphaZeroNet_Quantized'
        # Could be a quantized model without clear naming
        return 'Unknown_Quantized'
    
    if isinstance(weights, dict):
        # Check state dict keys to determine model type
        state_dict = weights.get('model_state_dict', weights)
        
        # TitanMini has specific modules like chess_positional_encoding and relative_position_bias
        has_chess_pos_encoding = any('chess_positional_encoding' in key for key in state_dict.keys())
        has_relative_pos_bias = any('relative_position_bias' in key for key in state_dict.keys())
        has_geglu = any('geglu' in key.lower() for key in state_dict.keys())
        has_cls_token = any('cls_token' in key for key in state_dict.keys())
        
        # PieBotNet has specific modules like positional_encoding and transformer_blocks
        has_positional_encoding = any('positional_encoding' in key for key in state_dict.keys())
        has_transformer = any('transformer_blocks' in key for key in state_dict.keys())
        
        # PieNano models have SE (Squeeze-Excitation) modules and depthwise convolutions
        has_se = any('se.' in key or 'squeeze' in key or 'excitation' in key for key in state_dict.keys())
        has_depthwise = any('depthwise' in key for key in state_dict.keys())
        has_wdl_value = any('value_head.fc2' in key for key in state_dict.keys())
        
        # PieNanoV2 has the improved policy head with fc1 and fc2 in policy_head
        has_improved_policy = any('policy_head.fc1' in key or 'policy_head.fc2' in key for key in state_dict.keys())
        
        if has_chess_pos_encoding or has_relative_pos_bias or has_geglu or has_cls_token:
            return 'TitanMini'
        elif has_positional_encoding or has_transformer:
            return 'PieBotNet'
        elif has_improved_policy and (has_se or has_depthwise):
            return 'PieNanoV2'
        elif (has_se or has_depthwise) and has_wdl_value:
            return 'PieNano'
    
    return 'AlphaZeroNet'


def clean_state_dict(state_dict):
    """
    Clean state dict by removing torch.compile prefixes and handling various formats.
    
    Args:
        state_dict: Model state dictionary
    
    Returns:
        dict: Cleaned state dictionary
    """
    # Handle _orig_mod prefix from torch.compile
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v  # Remove '_orig_mod.' prefix
            else:
                new_state_dict[k] = v
        return new_state_dict
    return state_dict


def create_titanmini_from_weights(weights):
    """
    Create TitanMini model with correct architecture from weights.
    Supports dynamic detection of any TitanMini configuration.
    
    Args:
        weights: Model checkpoint dictionary or state dict
    
    Returns:
        TitanMini model instance
    """
    # New default configuration (200MB model)
    DEFAULT_NUM_LAYERS = 13
    DEFAULT_D_MODEL = 512
    DEFAULT_NUM_HEADS = 8
    DEFAULT_D_FF = 1920
    
    # Initialize with defaults
    num_layers = DEFAULT_NUM_LAYERS
    d_model = DEFAULT_D_MODEL
    num_heads = DEFAULT_NUM_HEADS
    d_ff = DEFAULT_D_FF
    dropout = 0.1
    policy_weight = 1.0
    input_planes = 112
    use_wdl = True
    legacy_value_head = False
    
    if isinstance(weights, dict):
        # First priority: Check for saved model_config
        if 'model_config' in weights:
            config = weights['model_config']
            num_layers = config.get('num_layers', DEFAULT_NUM_LAYERS)
            d_model = config.get('d_model', DEFAULT_D_MODEL)
            num_heads = config.get('num_heads', DEFAULT_NUM_HEADS)
            d_ff = config.get('d_ff', DEFAULT_D_FF)
            dropout = config.get('dropout', 0.1)
            policy_weight = config.get('policy_weight', 1.0)
            input_planes = config.get('input_planes', 112)
            use_wdl = config.get('use_wdl', True)
            legacy_value_head = config.get('legacy_value_head', False)
        
        # Second priority: Check for args from training
        elif 'args' in weights:
            args = weights['args']
            if hasattr(args, 'num_layers'):
                num_layers = args.num_layers
            if hasattr(args, 'd_model'):
                d_model = args.d_model
            if hasattr(args, 'num_heads'):
                num_heads = args.num_heads
            if hasattr(args, 'd_ff'):
                d_ff = args.d_ff
            if hasattr(args, 'dropout'):
                dropout = args.dropout
            if hasattr(args, 'policy_weight'):
                policy_weight = args.policy_weight
            if hasattr(args, 'input_planes'):
                input_planes = args.input_planes
        
        # Third priority: Infer from state dict shapes
        else:
            state_dict = weights.get('model_state_dict', weights)
            
            # Clean keys by removing _orig_mod prefix
            clean_dict = {}
            for k, v in state_dict.items():
                clean_k = k.replace('_orig_mod.', '')
                clean_dict[clean_k] = v
            
            # 1. Infer d_model from input_projection
            if 'input_projection.weight' in clean_dict:
                d_model = clean_dict['input_projection.weight'].shape[0]
                input_planes = clean_dict['input_projection.weight'].shape[1]
            
            # 2. Infer num_layers by counting transformer blocks
            layer_indices = set()
            for k in clean_dict.keys():
                if 'transformer_blocks.' in k:
                    parts = k.split('.')
                    if len(parts) >= 2 and parts[1].isdigit():
                        layer_indices.add(int(parts[1]))
            if layer_indices:
                num_layers = max(layer_indices) + 1  # 0-indexed
            
            # 3. Infer num_heads from relative_position_bias
            for k, v in clean_dict.items():
                if 'relative_bias_table' in k:
                    # Shape is (15, 15, num_heads)
                    if len(v.shape) == 3:
                        num_heads = v.shape[2]
                        break
            
            # 4. Infer d_ff from GEGLU layer
            for k, v in clean_dict.items():
                if 'ffn.proj.weight' in k:
                    # GEGLU proj maps d_model -> 2*d_ff
                    # v.shape is (2*d_ff, d_model)
                    d_ff = v.shape[0] // 2
                    break
            
            # 5. Detect WDL mode
            use_wdl = False
            for k, v in clean_dict.items():
                if 'value_head.value_proj' in k and 'weight' in k:
                    # Check final layer output dimension
                    if '6.weight' in k:  # Final layer in WDL head
                        use_wdl = (v.shape[0] == 3)
                        break
            
            # 6. Detect legacy value head
            legacy_value_head = False
            if not use_wdl:
                # Check if value head has only 2 layers (legacy) or 3 layers (modern)
                has_fc3 = any('value_head.fc3' in k for k in clean_dict.keys())
                legacy_value_head = not has_fc3
    
    # Validate num_heads divides d_model evenly
    if d_model % num_heads != 0:
        # Adjust num_heads to nearest valid value
        valid_heads = [h for h in [4, 6, 8, 12, 16] if d_model % h == 0]
        if valid_heads:
            num_heads = min(valid_heads, key=lambda x: abs(x - num_heads))
        else:
            num_heads = 8  # Fallback
    
    # Create model with detected/configured parameters
    model = TitanMiniNetwork.TitanMini(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        policy_weight=policy_weight,
        input_planes=input_planes,
        use_wdl=use_wdl,
        legacy_value_head=legacy_value_head
    )
    
    print(f"Created TitanMini: layers={num_layers}, d_model={d_model}, "
          f"heads={num_heads}, d_ff={d_ff}, WDL={use_wdl}")
    
    return model


def create_pienano_from_weights(weights):
    """
    Create PieNano model with correct architecture from weights.
    
    Args:
        weights: Model checkpoint dictionary
    
    Returns:
        PieNano or PieNanoV2 model instance
    """
    # Try to detect architecture from weights
    if isinstance(weights, dict):
        # Check if it has args from training
        if 'args' in weights:
            args = weights['args']
            num_blocks = getattr(args, 'num_blocks', 8)
            num_filters = getattr(args, 'num_filters', 128)
            num_input_planes = 112 if getattr(args, 'use_enhanced_encoder', False) else 16
            policy_hidden_dim = getattr(args, 'policy_hidden_dim', None)
        else:
            # Try to infer from state dict
            state_dict = weights.get('model_state_dict', weights)
            
            # Count residual blocks
            residual_keys = [k for k in state_dict.keys() if 'residual_tower' in k and 'conv1' in k]
            num_blocks = len(set(k.split('.')[1] for k in residual_keys if len(k.split('.')) > 1))
            
            # Get number of filters and input planes from conv_block
            if 'conv_block.0.weight' in state_dict:
                num_filters = state_dict['conv_block.0.weight'].shape[0]
                num_input_planes = state_dict['conv_block.0.weight'].shape[1]
            else:
                # Default values
                num_filters = 128
                num_input_planes = 16
            
            # Detect if it's V2 by checking for policy_head.fc1
            policy_hidden_dim = None
            if 'policy_head.fc1.weight' in state_dict:
                policy_hidden_dim = state_dict['policy_head.fc1.weight'].shape[0]
            
            # Default to 8 blocks if detection failed
            if num_blocks == 0:
                num_blocks = 8
    else:
        # Default PieNano configuration
        num_blocks = 8
        num_filters = 128
        num_input_planes = 16
        policy_hidden_dim = None
    
    # Create V2 if policy_hidden_dim is detected, otherwise V1
    if policy_hidden_dim is not None:
        return PieNanoNetwork_v2.PieNanoV2(
            num_blocks=num_blocks,
            num_filters=num_filters,
            num_input_planes=num_input_planes,
            policy_hidden_dim=policy_hidden_dim
        )
    else:
        # Fallback to V1 for older models
        return PieNanoNetwork.PieNano(
            num_blocks=num_blocks,
            num_filters=num_filters,
            num_input_planes=num_input_planes
        )


def load_quantized_titanmini(model_path):
    """
    Load a quantized TitanMini model from a state dict file.
    Automatically detects the model architecture from the weights.
    
    Args:
        model_path: Path to the quantized model file
    
    Returns:
        Quantized TitanMini model (runs on CPU)
    """
    # Load the full checkpoint (may contain metadata)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Create a TitanMini model with detected architecture
    # The create_titanmini_from_weights function will handle detection
    model = create_titanmini_from_weights(checkpoint)
    
    # Move to CPU (quantized models must run on CPU)
    model = model.cpu()
    model.eval()
    
    # Get the state dict from checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Filter out quantization-specific keys and load regular weights
    regular_state_dict = {}
    for k, v in state_dict.items():
        if not any(x in k for x in ['scale', 'zero_point', '_packed_params']):
            regular_state_dict[k] = v
    
    # Clean state dict
    regular_state_dict = clean_state_dict(regular_state_dict)
    
    # Try to load what we can
    model.load_state_dict(regular_state_dict, strict=False)
    
    # Apply dynamic quantization to recreate the quantized model
    torch.backends.quantized.engine = 'qnnpack'
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={torch.nn.Linear},
        dtype=torch.qint8
    )
    
    for param in quantized_model.parameters():
        param.requires_grad = False
    
    return quantized_model


def get_titanmini_config_from_model(model_path):
    """
    Extract TitanMini configuration from a saved model.
    Useful for quantization scripts that need to know the architecture.
    
    Args:
        model_path: Path to model file
    
    Returns:
        dict: Configuration dictionary with architecture parameters
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Use create_titanmini_from_weights logic but return config instead of model
    config = {
        'num_layers': 13,  # Default
        'd_model': 512,
        'num_heads': 8,
        'd_ff': 1920,
        'dropout': 0.1,
        'policy_weight': 1.0,
        'input_planes': 112,
        'use_wdl': True,
        'legacy_value_head': False
    }
    
    if isinstance(checkpoint, dict):
        # Check for saved model_config
        if 'model_config' in checkpoint:
            config.update(checkpoint['model_config'])
        # Check for args
        elif 'args' in checkpoint:
            args = checkpoint['args']
            for key in config.keys():
                if hasattr(args, key):
                    config[key] = getattr(args, key)
        # Infer from state dict
        else:
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            clean_dict = {}
            for k, v in state_dict.items():
                clean_k = k.replace('_orig_mod.', '')
                clean_dict[clean_k] = v
            
            # Infer parameters as in create_titanmini_from_weights
            if 'input_projection.weight' in clean_dict:
                config['d_model'] = clean_dict['input_projection.weight'].shape[0]
                config['input_planes'] = clean_dict['input_projection.weight'].shape[1]
            
            # Count layers
            layer_indices = set()
            for k in clean_dict.keys():
                if 'transformer_blocks.' in k:
                    parts = k.split('.')
                    if len(parts) >= 2 and parts[1].isdigit():
                        layer_indices.add(int(parts[1]))
            if layer_indices:
                config['num_layers'] = max(layer_indices) + 1
            
            # Get num_heads
            for k, v in clean_dict.items():
                if 'relative_bias_table' in k and len(v.shape) == 3:
                    config['num_heads'] = v.shape[2]
                    break
            
            # Get d_ff
            for k, v in clean_dict.items():
                if 'ffn.proj.weight' in k:
                    config['d_ff'] = v.shape[0] // 2
                    break
    
    return config


def load_model(model_path, device=None):
    """
    Load any supported chess model with automatic detection.
    
    Args:
        model_path: Path to model file
        device: Optional device to load model on
    
    Returns:
        tuple: (model, device, is_quantized)
    """
    from device_utils import get_optimal_device, optimize_for_device
    
    if device is None:
        device, device_str = get_optimal_device()
    else:
        device_str = str(device)
    
    # Try loading as TorchScript first (for quantized models)
    is_quantized = False
    try:
        # Attempt to load as TorchScript
        model = torch.jit.load(model_path, map_location='cpu')
        is_quantized = True
        model.eval()
        print(f"Loaded quantized model (TorchScript) on CPU")
        device = torch.device('cpu')  # Quantized models run on CPU
        for param in model.parameters():
            param.requires_grad = False
        return model, device, is_quantized
    except:
        # Not a TorchScript file, continue with regular loading
        pass
    
    # Load weights to check model type
    weights = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Detect and create the appropriate model type
    model_type = detect_model_type(weights, model_path)
    
    if model_type == 'TitanMini_Quantized':
        # Handle quantized TitanMini
        try:
            # Check if it's a TorchScript quantized model
            if model_path.endswith('.jit') or model_path.endswith('.pt'):
                try:
                    # Try loading as TorchScript first
                    model = torch.jit.load(model_path, map_location='cpu')
                    is_quantized = True
                    model.eval()
                    print(f"Loaded quantized TitanMini model (TorchScript) on CPU")
                    device = torch.device('cpu')
                    for param in model.parameters():
                        param.requires_grad = False
                    return model, device, is_quantized
                except:
                    # Not TorchScript, try dynamic quantization approach
                    pass
            
            # Use dynamic quantization approach
            model = load_quantized_titanmini(model_path)
            is_quantized = True
            print(f"Loaded quantized TitanMini model on CPU")
            device = torch.device('cpu')  # Quantized models run on CPU
            return model, device, is_quantized
        except Exception as e:
            print(f"Warning: Failed to load as quantized TitanMini: {e}")
            # Fallback: create regular TitanMini
            model = create_titanmini_from_weights(weights)
            print(f"Loading TitanMini model (dequantized fallback) on {device_str}")
    elif model_type == 'TitanMini':
        model = create_titanmini_from_weights(weights)
        print(f"Loading TitanMini model on {device_str}")
    elif model_type == 'AlphaZeroNet_Quantized':
        # Handle quantized AlphaZeroNet
        # AlphaZeroNet quantized models are saved as TorchScript, which should be loaded above
        # If we reach here, it means the TorchScript loading failed
        print(f"Note: AlphaZeroNet quantized model should be loaded as TorchScript")
        # Fallback to regular AlphaZeroNet
        model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
        print(f"Loading AlphaZeroNet model (dequantized fallback) on {device_str}")
    elif model_type == 'PieNano_Quantized':
        # Try to load quantized PieNano
        try:
            from quantization_utils import load_quantized_model
            model = load_quantized_model(model_path)
            is_quantized = True
            print(f"Loaded quantized PieNano model on CPU")
            device = torch.device('cpu')
            return model, device, is_quantized
        except:
            # Fallback: create regular PieNano
            model = create_pienano_from_weights(weights)
            print(f"Loading PieNano model (dequantized fallback) on {device_str}")
    elif model_type == 'PieBotNet':
        model = PieBotNetwork.PieBotNet()
        print(f"Loading PieBotNet model on {device_str}")
    elif model_type == 'PieNanoV2':
        model = create_pienano_from_weights(weights)
        print(f"Loading PieNanoV2 model on {device_str}")
    elif model_type == 'PieNano':
        model = create_pienano_from_weights(weights)
        print(f"Loading PieNano model on {device_str}")
    else:
        # Default to AlphaZeroNet
        model = AlphaZeroNetwork.AlphaZeroNet(20, 256)
        print(f"Loading AlphaZeroNet model on {device_str}")
    
    # Only load state dict if not quantized
    if not is_quantized:
        # Handle different model formats
        if isinstance(weights, dict) and 'model_state_dict' in weights:
            state_dict = weights['model_state_dict']
        else:
            state_dict = weights
        
        # Clean state dict (remove _orig_mod prefix etc.)
        state_dict = clean_state_dict(state_dict)
        
        model.load_state_dict(state_dict)
        
        # Handle FP16 models
        if isinstance(weights, dict) and weights.get('model_type') == 'fp16':
            model = model.half()
            print(f"Loaded FP16 model on {device_str}")
        
        model = optimize_for_device(model, device)
    
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model, device, is_quantized