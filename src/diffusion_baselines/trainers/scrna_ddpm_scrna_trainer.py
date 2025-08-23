"""
ScRNATrainer: Extends a generic DDPMTrainer to handle paired scRNA-seq data.
Performs conditional diffusion training for Control → IFN perturbation mapping.
"""

import os
import torch
from tqdm import tqdm

from .ddpm_trainer import DDPMTrainer

class ScRNATrainer(DDPMTrainer):
    """
    Trainer for conditional scRNA-seq diffusion.
    
    This class extends DDPMTrainer to:
      - Accept paired input (control, IFN) RNA-seq vectors.
      - Compute conditional diffusion loss leveraging GaussianDiffusionTrainer.
      - Save model checkpoints at configured intervals.
    """
    def __init__(self,
                 model: torch.nn.Module,              # The conditional DDPM network
                 diffusion_trainer: torch.nn.Module,  # Underlying GaussianDiffusionTrainer
                 optimizer: torch.optim.Optimizer,
                 scheduler,                          # Learning rate scheduler
                 loader: torch.utils.data.DataLoader,
                 device: torch.device,
                 cfg):
        """
        Initialize the scRNA diffusion trainer.

        Args:
            model: The DDPM model predicting noise given x_t and control embedding.
            diffusion_trainer: Instance of GaussianDiffusionTrainer for q-sampling.
            optimizer: Optimizer instance (e.g., AdamW) for model parameters.
            scheduler: Learning rate scheduler with optional warmup.
            loader: DataLoader yielding (control, IFN) batches.
            device: Device (cuda or cpu) to run training on.
            cfg: Configuration object containing training parameters.
        """
        # Initialize base trainer with core training components
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loader=loader,
            device=device,
            cfg=cfg
        )
        # Store the diffusion trainer on the correct device for noise scheduling
        self.diff_trainer = diffusion_trainer.to(device)

    def compute_loss(self, x_ctrl: torch.Tensor, x_ifn: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean squared error loss for one batch of (control, IFN) data.

        Steps:
        1. Sample a random timestep t for each example.
        2. Add noise to IFN embedding (x_ifn) to obtain x_t.
        3. Predict noise with the model conditioned on control embedding (x_ctrl).
        4. Compute MSE loss between the predicted noise and the original noise.

        Args:
            x_ctrl: Control condition tensor of shape [B, D].
            x_ifn: IFN condition tensor of shape [B, D].

        Returns:
            Scalar tensor representing the batch MSE loss.
        """
        # Move inputs to the configured device
        x = x_ctrl.to(self.device)  # Control embedding [B, D]
        y = x_ifn.to(self.device)   # IFN embedding [B, D]

        # Sample random timesteps uniformly from [0, T)
        B = x.size(0)
        T = self.diff_trainer.T
        t = torch.randint(0, T, (B,), device=self.device).long()

        # Generate standard Gaussian noise
        noise = torch.randn_like(x)

        # Extract precomputed coefficients for the forward diffusion process
        # sqrt_alphas_bar[t] and sqrt_one_minus_alphas_bar[t] are 1D buffers of length T
        a_bar = self.diff_trainer.sqrt_alphas_bar[t].view(B, 1)        # [B, 1]
        b_bar = self.diff_trainer.sqrt_one_minus_alphas_bar[t].view(B, 1) # [B, 1]

        # Create the noisy observation x_t = a_bar * y + b_bar * noise
        x_t = a_bar * y + b_bar * noise  # [B, D]

        # Model predicts noise given x_t and control embedding
        pred_noise = self.model(x_t, x, t)  # [B, D]

        # ### 关键改动 1: 简化损失计算以提高数值稳定性 ###
        # 原始代码: true_noise = (x_t - a_bar * y) / b_bar
        # 这个除法在 b_bar 接近 0 时（即 t 接近 0 时）可能导致数值不稳定。
        # 事实上，我们的目标就是让模型预测出我们一开始加入的 `noise`。
        # 因此，直接使用 `noise` 作为目标是最直接和最稳定的方法。
        loss = torch.nn.functional.mse_loss(pred_noise, noise)
        return loss

    def train(self):
        """
        Execute the training loop over configured epochs.
        - Iterates over epochs and batches.
        - Computes loss and updates model parameters.
        - Saves checkpoints at specified intervals.
        """
        # Ensure save directory exists
        save_dir = self.cfg.train.save_weight_dir
        os.makedirs(save_dir, exist_ok=True)

        num_epochs = self.cfg.train.epoch
        ckpt_interval = self.cfg.train.ckpt_save_interval
        grad_clip_norm = self.cfg.train.get('grad_clip_norm', 1.0) # 从配置获取裁剪范数，默认为1.0

        # Outer loop: epochs
        for epoch in range(num_epochs):
            # Inner loop: batches with tqdm progress bar
            batch_iter = tqdm(self.loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
            for x_ctrl, x_ifn in batch_iter:
                # Compute loss for current batch
                loss = self.compute_loss(x_ctrl, x_ifn)

                # Backpropagation and parameter update
                self.optimizer.zero_grad()
                loss.backward()

                # ### 关键改动 2: 增加梯度裁剪 ###
                # 这是防止梯度爆炸导致 NaN 的核心步骤。
                # 它会将所有参数的梯度范数限制在一个最大值（这里是 grad_clip_norm）。
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)
                
                self.optimizer.step()

                # Update progress bar with current loss
                batch_iter.set_postfix({"loss": f"{loss.item():.4f}"})

            # Step the learning rate scheduler after each epoch
            self.scheduler.step()

            # Save checkpoint if at the specified interval
            if (epoch + 1) % ckpt_interval == 0:
                ckpt_path = os.path.join(save_dir, f"scrna_ddpm_epoch{epoch+1}.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

