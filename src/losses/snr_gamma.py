import torch
import torch.nn.functional as F

from diffusers.training_utils import compute_snr

def masked_mse(x, y, mask=None):
    if mask is not None:
        loss = ((x - y) * mask) ** 2
        non_zero = mask.sum(dim=list(range(1, len(mask.shape)))) * x.shape[1]
        loss = loss.sum(dim=list(range(1, len(loss.shape)))) / non_zero
    else:
        loss = (x - y) ** 2
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
    return loss



class SNRGammaLoss(torch.nn.Module):
    def __init__(self, snr_gamma: float, mask_gamma=None):
        super().__init__()
        self.snr_gamma = snr_gamma
        self.mask_gamma = mask_gamma

    def forward(self, model_pred, target, noise_scheduler, timesteps, mask=None, ):
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack([snr, self.snr_gamma * \
            torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)
        else: 
            raise ValueError("Unknown prediction type")
        loss = masked_mse(model_pred, target, mask)
        loss = loss * mse_loss_weights
        loss = loss.mean()
        if self.mask_gamma and mask:
            loss_m = masked_mse(model_pred, target, 1 - mask)
            loss_m = loss_m * mse_loss_weights
            loss_m = loss_m.mean()
            loss = loss * self.mask_gamma + loss_m * (1 - self.mask_gamma)
        return loss