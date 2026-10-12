import torch
from pytorch_wavelets import DWTForward


class WaveletsLoss(torch.nn.Module):
    def __init__(
        self,
        ll_weight: float = 0.01,
        lh_weight: float = 0.03,
        hl_weight: float = 0.03,
        hh_weight: float = 0.05,
    ):
        super().__init__()
        self.dwt = DWTForward(J=1, mode="reflect", wave="haar")

        # Shape [1, 4, 1, 1, 1] to broadcast over [B, 4, C, H, W]
        weights = torch.tensor(
            [ll_weight, lh_weight, hl_weight, hh_weight], dtype=torch.float32
        )
        self.register_buffer("weights", weights.view(1, 4, 1, 1, 1))

    def forward(self, model_pred, target):
        with torch.no_grad():
            target_xll, target_xh = self.dwt(target)
            target_xlh, target_xhl, target_xhh = torch.unbind(
                target_xh[0], dim=2
            )
            target = torch.stack(
                [target_xll, target_xlh, target_xhl, target_xhh], dim=1
            )

        model_pred_xll, model_pred_xh = self.dwt(model_pred)
        model_pred_xlh, model_pred_xhl, model_pred_xhh = torch.unbind(
            model_pred_xh[0], dim=2
        )
        model_pred = torch.stack(
            [model_pred_xll, model_pred_xlh, model_pred_xhl, model_pred_xhh], dim=1
        )

        loss = torch.nn.functional.huber_loss(model_pred, target, reduction="none")
        loss = loss * self.weights
        return loss.sum(dim=1)