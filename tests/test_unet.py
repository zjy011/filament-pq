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
