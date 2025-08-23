# trainers/mlp_ddpm_mlp_trainer.py

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

class ScRNATrainer:
    """
    用于 encoder→DDPM→decoder 模型的训练器。
    数据加载器 (Loader) 的每个批次返回 (x0, x1)，分别代表扰动前和扰动后的 scRNA 数据。
    """
    def __init__(self, model: torch.nn.Module, diffusion, optimizer, scheduler, data_loader: DataLoader, device: torch.device, cfg):
        """
        初始化训练器。

        :param model: 要训练的 MLPDDPMMLP 模型。
        :param diffusion: GaussianDiffusionTrainer 实例 (已包含在模型中，此处为显式传入)。
        :param optimizer: 优化器 (例如, AdamW)。
        :param scheduler: 学习率调度器。
        :param data_loader: 训练数据的 DataLoader。
        :param device: torch 设备 ('cuda' 或 'cpu')。
        :param cfg: OmegaConf 配置对象。
        """
        self.model = model
        self.diffusion = diffusion  # 注意: model.forward 内部会使用自己的 self.diffusion_trainer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loader = data_loader
        self.device = device
        self.cfg = cfg
        
        self.train_cfg = cfg.train
        self.save_dir = self.train_cfg.save_weight_dir
        
        # 如果指定了保存目录，则创建它
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            
        self.current_epoch = 0
        self.current_step = 0

    def compute_loss(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        计算模型的损失。

        :param x0: 扰动前 scRNA 数据 [批大小, 基因数量]。
        :param x1: 扰动后 scRNA 数据 [批大小, 基因数量]。
        :return: 一个标量损失值 (由模型的 forward 方法直接返回)。
        """
        # 将数据移动到指定设备
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)
        
        # 模型的前向传播直接计算并返回 DDPM 的训练损失
        return self.model(x0, x1)

    def train(self):
        """
        执行完整的训练流程。
        """
        print("开始训练...")
        self.model.train()  # 将模型设置为训练模式
        
        # 按设定的总轮数进行迭代
        for epoch in range(self.current_epoch, self.train_cfg.epoch):
            self.current_epoch = epoch
            # 使用 tqdm 显示带进度条的迭代过程
            progress_bar = tqdm(self.loader, desc=f"第 {epoch+1}/{self.train_cfg.epoch} 轮")
            
            for x0, x1 in progress_bar:
                # 1. 清空梯度
                self.optimizer.zero_grad()
                
                # 2. 计算损失
                loss = self.compute_loss(x0, x1)
                
                # 3. 反向传播
                loss.backward()
                
                # 4. 更新模型参数
                self.optimizer.step()
                
                self.current_step += 1
                
                # 5. 在进度条上显示实时损失和学习率
                progress_bar.set_postfix({
                    "损失": f"{loss.item():.4f}",
                    "学习率": f"{self.scheduler.get_last_lr()[0]:.6f}"
                })

            # 每轮结束后更新学习率
            self.scheduler.step()

            # 根据配置决定是否保存检查点
            if self.save_dir and (epoch + 1) % self.train_cfg.ckpt_save_interval == 0:
                self.save_checkpoint(epoch)
                
        print("训练完成。")

    def save_checkpoint(self, epoch: int):
        """
        保存模型检查点，包括模型权重、优化器和调度器状态。
        """
        ckpt_path = os.path.join(self.save_dir, f"model_epoch_{epoch+1}.pth")
        
        # 将训练状态打包成一个字典进行保存
        torch.save({
            'epoch': epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, ckpt_path)
        
        print(f"检查点已保存至: {ckpt_path}")