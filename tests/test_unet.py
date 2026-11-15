import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")


def test_unet_forward_backward_without_weight_download():
    from filament.unet.model import build_model, dice_bce

    torch.set_num_threads(2)
    model = build_model("resnet18", weights=None)
    logits = model(torch.rand(2, 1, 64, 64))
    assert logits.shape == (2, 1, 64, 64)
    loss = dice_bce(logits, torch.zeros_like(logits))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_cldice_rewards_continuity_and_backpropagates():
    from filament.unet.cldice import soft_cldice

    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 15:18, 4:28] = 1
    assert float(soft_cldice(target, target)) == 0
    broken = target.clone()
    broken[:, :, :, 12:20] = 0
    broken.requires_grad_()
    loss = soft_cldice(broken, target)
    assert 0 < float(loss.detach()) < 1
    loss.backward()
    assert torch.isfinite(broken.grad).all()
    assert broken.grad.abs().sum() > 0
    assert float(soft_cldice(torch.zeros_like(target), torch.zeros_like(target))) == 0
