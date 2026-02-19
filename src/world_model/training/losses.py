import torch


def mse_prediction_loss(pred_future: torch.Tensor, target_future: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred_future - target_future) ** 2)
