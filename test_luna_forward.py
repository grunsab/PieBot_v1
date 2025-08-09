import torch
from LunaNetwork import Luna


def test_forward_16_planes():
    model = Luna(input_planes=16, use_rope=True, use_alibi=True)
    x = torch.randn(2, 16, 8, 8)
    v_t = torch.randn(2, 1)
    p_t = torch.randint(0, 4608, (2,))
    total, v, p = model(x, v_t, p_t)
    assert total.shape == (), 'loss should be scalar (0-dim tensor)'


def test_forward_112_planes_mask_and_dist():
    model = Luna(input_planes=112)
    x = torch.randn(2, 112, 8, 8)
    v_t = torch.randn(2, 1)
    p_t = torch.rand(2, 4608)
    mask = torch.randint(0, 2, (2, 72, 8, 8))
    total, v, p = model(x, v_t, p_t, mask)
    assert total.ndim == 0


def test_inference_masked_softmax():
    model = Luna()
    x = torch.randn(1, 16, 8, 8)
    mask = torch.zeros(1, 72, 8, 8)
    # allow a single move index 0
    mask = mask.view(1, -1)
    mask[0, 0] = 1
    mask = mask.view(1, 72, 8, 8)
    with torch.no_grad():
        v, policy = model(x, policyMask=mask)
    assert policy.shape[-1] == 4608
    assert torch.allclose(policy.sum(dim=1), torch.ones(1), atol=1e-5)
    assert torch.argmax(policy, dim=1).item() == 0
