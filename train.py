import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import json
import numpy as np
from tqdm import tqdm
import wandb
from dataset.ego_dataset import EgoDataset
from model.crossmodal_net import crossmodal_net
from config.config import MotionFromGazeConfig
import argparse

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize wandb
        wandb.init(
            project=config.wandb_project,
            config=vars(config),
            name=f"GIMO_{time.strftime('%Y%m%d_%H%M%S')}",
            entity=config.wandb_entity,
            mode=config.wandb_mode
        )
        
        # Create datasets
        self.train_dataset = EgoDataset(config, train=True)
        self.valid_dataset = EgoDataset(config, train=False)
        
        # Create dataloaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True
        )
        
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False
        )
        
        # Create model
        if config.model_type == 'cross':
            self.model = crossmodal_net(config)
        else:
            raise NotImplementedError(f"Model type {config.model_type} not implemented")
        
        self.model = self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Create learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=config.gamma
        )
        
        # Load checkpoint if specified
        if config.load_model_dir is not None:
            print(f"Loading model from {config.load_model_dir}")
            self.model.load_state_dict(torch.load(config.load_model_dir))
            
        if config.load_optim_dir is not None:
            print(f"Loading optimizer from {config.load_optim_dir}")
            self.optimizer.load_state_dict(torch.load(config.load_optim_dir))
        
        # Create save directory if it doesn't exist
        os.makedirs(config.save_path, exist_ok=True)
        
        # Create log file
        self.log_file = os.path.join(config.save_path, 'train_log.txt')
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Config: {config.__dict__}\n\n")
    
    def log(self, message):
        """Write message to log file and print to console"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")
    
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        epoch_trans_loss = 0
        epoch_ori_loss = 0
        epoch_latent_loss = 0
        epoch_rec_loss = 0
        epoch_des_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for i, data in enumerate(progress_bar):
            gazes, gazes_mask, poses_input, smplx_vertices, poses_label, poses_mask, scene_points = data
            
            # Move data to device
            gazes = gazes.to(self.device)
            gazes_mask = gazes_mask.to(self.device)
            poses_input = poses_input.to(self.device)
            smplx_vertices = smplx_vertices.to(self.device)
            poses_label = poses_label.to(self.device)
            poses_mask = poses_mask.to(self.device)
            scene_points = scene_points.to(self.device).contiguous()
            
            # Forward pass
            self.optimizer.zero_grad()
            poses_predict = self.model(gazes, gazes_mask, poses_input, smplx_vertices, scene_points)
            
            # Calculate losses
            # Destination loss
            loss_des_ori = F.l1_loss(poses_predict[:, -1, :3], poses_label[:, -1, :3])
            loss_des_trans = F.l1_loss(poses_predict[:, -1, 3:6], poses_label[:, -1, 3:6])
            loss_des_latent = F.l1_loss(poses_predict[:, -1, 6:], poses_label[:, -1, 6:])
            
            # Path loss
            loss_all = F.l1_loss(poses_predict[:, self.config.input_seq_len:-1, :], poses_label[:, :-1], reduction='none')
            loss_all *= poses_mask[:, :-1].unsqueeze(2)  # Apply mask
            
            # Reconstruction loss
            loss_rec = F.l1_loss(poses_predict[:, :self.config.input_seq_len, :], poses_input)
            
            # Calculate mean losses for path
            loss_ori = (loss_all[:, :, :3].sum(dim=1) / poses_mask[:, :-1].sum(dim=1, keepdim=True)).mean()
            loss_trans = (loss_all[:, :, 3:6].sum(dim=1) / poses_mask[:, :-1].sum(dim=1, keepdim=True)).mean()
            loss_latent = (loss_all[:, :, 6:].sum(dim=1) / poses_mask[:, :-1].sum(dim=1, keepdim=True)).mean()
            
            # Weighted total loss
            loss_des = self.config.lambda_ori * loss_des_ori + \
                       self.config.lambda_trans * loss_des_trans + \
                       self.config.lambda_latent * loss_des_latent
            
            loss = self.config.lambda_ori * loss_ori + \
                   self.config.lambda_trans * loss_trans + \
                   self.config.lambda_latent * loss_latent + \
                   self.config.lambda_rec * loss_rec + \
                   self.config.lambda_des * loss_des
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'ori': loss_ori.item(),
                'trans': loss_trans.item(),
                'rec': loss_rec.item(),
                'des': loss_des.item()
            })
            
            # Update epoch losses
            epoch_loss += loss.item()
            epoch_trans_loss += loss_trans.item()
            epoch_ori_loss += loss_ori.item()
            epoch_latent_loss += loss_latent.item()
            epoch_rec_loss += loss_rec.item()
            epoch_des_loss += loss_des.item()
            
            # Visualize if needed
            if (i + 1) % self.config.vis_fre == 0:
                self.log(f"Epoch {epoch} Iter {i+1}/{len(self.train_loader)}: "
                         f"Loss {loss.item():.4f}, Ori {loss_ori.item():.4f}, "
                         f"Trans {loss_trans.item():.4f}, Rec {loss_rec.item():.4f}, "
                         f"Des {loss_des.item():.4f}")
        
        # Calculate average epoch losses
        epoch_loss /= len(self.train_loader)
        epoch_trans_loss /= len(self.train_loader)
        epoch_ori_loss /= len(self.train_loader)
        epoch_latent_loss /= len(self.train_loader)
        epoch_rec_loss /= len(self.train_loader)
        epoch_des_loss /= len(self.train_loader)
        
        # Log epoch metrics to wandb
        wandb.log({
            'train/epoch_loss': epoch_loss,
            'train/epoch_trans_loss': epoch_trans_loss,
            'train/epoch_ori_loss': epoch_ori_loss,
            'train/epoch_latent_loss': epoch_latent_loss,
            'train/epoch_rec_loss': epoch_rec_loss,
            'train/epoch_des_loss': epoch_des_loss,
            'epoch': epoch
        })
        
        return {
            'loss': epoch_loss,
            'trans_loss': epoch_trans_loss,
            'ori_loss': epoch_ori_loss,
            'latent_loss': epoch_latent_loss,
            'rec_loss': epoch_rec_loss,
            'des_loss': epoch_des_loss
        }
    
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        val_trans_loss = 0
        val_ori_loss = 0
        val_latent_loss = 0
        val_rec_loss = 0
        val_des_loss = 0
        
        with torch.no_grad():
            for data in tqdm(self.valid_loader, desc=f"Validation Epoch {epoch}"):
                gazes, gazes_mask, poses_input, smplx_vertices, poses_label, poses_mask, scene_points = data
                
                # Move data to device
                gazes = gazes.to(self.device)
                gazes_mask = gazes_mask.to(self.device)
                poses_input = poses_input.to(self.device)
                smplx_vertices = smplx_vertices.to(self.device)
                poses_label = poses_label.to(self.device)
                poses_mask = poses_mask.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                
                # Forward pass
                poses_predict = self.model(gazes, gazes_mask, poses_input, smplx_vertices, scene_points)
                
                # Calculate losses
                loss_des_ori = F.l1_loss(poses_predict[:, -1, :3], poses_label[:, -1, :3])
                loss_des_trans = F.l1_loss(poses_predict[:, -1, 3:6], poses_label[:, -1, 3:6])
                loss_des_latent = F.l1_loss(poses_predict[:, -1, 6:], poses_label[:, -1, 6:])
                
                loss_all = F.l1_loss(poses_predict[:, self.config.input_seq_len:-1, :], poses_label[:, :-1], reduction='none')
                loss_rec = F.l1_loss(poses_predict[:, :self.config.input_seq_len, :], poses_input)
                loss_all *= poses_mask[:, :-1].unsqueeze(2)
                
                loss_ori = (loss_all[:, :, :3].sum(dim=1) / poses_mask[:, :-1].sum(dim=1, keepdim=True)).mean()
                loss_trans = (loss_all[:, :, 3:6].sum(dim=1) / poses_mask[:, :-1].sum(dim=1, keepdim=True)).mean()
                loss_latent = (loss_all[:, :, 6:].sum(dim=1) / poses_mask[:, :-1].sum(dim=1, keepdim=True)).mean()
                
                loss_des = self.config.lambda_ori * loss_des_ori + \
                           self.config.lambda_trans * loss_des_trans + \
                           self.config.lambda_latent * loss_des_latent
                
                loss = self.config.lambda_ori * loss_ori + \
                       self.config.lambda_trans * loss_trans + \
                       self.config.lambda_latent * loss_latent + \
                       self.config.lambda_rec * loss_rec + \
                       self.config.lambda_des * loss_des
                
                val_loss += loss.item()
                val_trans_loss += loss_trans.item()
                val_ori_loss += loss_ori.item()
                val_latent_loss += loss_latent.item()
                val_rec_loss += loss_rec.item()
                val_des_loss += loss_des.item()
        
        # Calculate average validation losses
        val_loss /= len(self.valid_loader)
        val_trans_loss /= len(self.valid_loader)
        val_ori_loss /= len(self.valid_loader)
        val_latent_loss /= len(self.valid_loader)
        val_rec_loss /= len(self.valid_loader)
        val_des_loss /= len(self.valid_loader)
        
        # Log validation metrics to wandb
        wandb.log({
            'val/loss': val_loss,
            'val/trans_loss': val_trans_loss,
            'val/ori_loss': val_ori_loss,
            'val/latent_loss': val_latent_loss,
            'val/rec_loss': val_rec_loss,
            'val/des_loss': val_des_loss,
            'epoch': epoch
        })
        
        return {
            'loss': val_loss,
            'trans_loss': val_trans_loss,
            'ori_loss': val_ori_loss,
            'latent_loss': val_latent_loss,
            'rec_loss': val_rec_loss,
            'des_loss': val_des_loss
        }
    
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config.epoch + 1):
            # Train one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Log training metrics
            self.log(f"Epoch {epoch} Training: Loss {train_metrics['loss']:.4f}, "
                     f"Trans {train_metrics['trans_loss']:.4f}, Ori {train_metrics['ori_loss']:.4f}, "
                     f"Latent {train_metrics['latent_loss']:.4f}, Rec {train_metrics['rec_loss']:.4f}, "
                     f"Des {train_metrics['des_loss']:.4f}")
            
            # Validate if needed
            if epoch % self.config.val_fre == 0:
                val_metrics = self.validate(epoch)
                
                self.log(f"Epoch {epoch} Validation: Loss {val_metrics['loss']:.4f}, "
                         f"Trans {val_metrics['trans_loss']:.4f}, Ori {val_metrics['ori_loss']:.4f}, "
                         f"Latent {val_metrics['latent_loss']:.4f}, Rec {val_metrics['rec_loss']:.4f}, "
                         f"Des {val_metrics['des_loss']:.4f}")
                
                # Save model if it's the best so far
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.log(f"New best validation loss: {best_val_loss:.4f}")
                    torch.save(self.model.state_dict(), os.path.join(self.config.save_path, 'best_model.pth'))
            
            # Save model checkpoint
            if epoch % self.config.save_fre == 0:
                torch.save(self.model.state_dict(), os.path.join(self.config.save_path, f'model_epoch_{epoch}.pth'))
                torch.save(self.optimizer.state_dict(), os.path.join(self.config.save_path, f'optim_epoch_{epoch}.pth'))
                self.log(f"Saved checkpoint at epoch {epoch}")
                
                # Save model to wandb
                wandb.save(os.path.join(self.config.save_path, f'model_epoch_{epoch}.pth'))
                wandb.save(os.path.join(self.config.save_path, f'optim_epoch_{epoch}.pth'))
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            self.log(f"Learning rate updated to {current_lr:.6f}")
            
            # Save final model
            if epoch == self.config.epoch:
                torch.save(self.model.state_dict(), os.path.join(self.config.save_path, 'final_model.pth'))
                torch.save(self.optimizer.state_dict(), os.path.join(self.config.save_path, 'final_optim.pth'))
                wandb.save(os.path.join(self.config.save_path, 'final_model.pth'))
                wandb.save(os.path.join(self.config.save_path, 'final_optim.pth'))
                
        # Finish wandb run
        wandb.finish()

if __name__ == '__main__':
    # Parse arguments
    config = MotionFromGazeConfig().parse_args()
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train() 