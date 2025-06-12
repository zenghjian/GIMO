import math 
import numpy as np
import os 
from tqdm.auto import tqdm
from einops import rearrange, reduce
from inspect import isfunction

import torch
from torch import nn
import torch.nn.functional as F

# Import geometry utilities from GIMO_ADT
from utils.geometry_utils import (
    rotation_matrix_to_6d,
    rotation_6d_to_matrix,
    rotation_6d_to_matrix_torch,
    rotation_matrix_to_6d_torch
)

# Import CHOIS Transformer Decoder
from model.chois_transformer_module import Decoder

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered



class TransformerDiffusionModel(nn.Module):
    def __init__(
        self,
        d_input_feats,
        d_feats,
        d_model,
        n_dec_layers,
        n_head,
        d_k,
        d_v,
        max_timesteps,
    ):
        super().__init__()
        
        self.d_feats = d_feats 
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k 
        self.d_v = d_v 
        self.max_timesteps = max_timesteps 

        # Use original CHOIS Decoder - Input: BS X D X T, Output: BS X T X D'
        self.motion_transformer = Decoder(
            d_feats=d_input_feats, 
            d_model=self.d_model,
            n_layers=self.n_dec_layers, 
            n_head=self.n_head, 
            d_k=self.d_k, 
            d_v=self.d_v,
            max_timesteps=self.max_timesteps, 
            use_full_attention=True
        )

        self.linear_out = nn.Linear(self.d_model, self.d_feats)

        # For noise level t embedding
        dim = 64
        learned_sinusoidal_dim = 16
        time_dim = dim * 4

        learned_sinusoidal_cond = False
        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model)
        )

    def forward(self, src, noise_t, condition, language_embedding=None, padding_mask=None):
        # src: BS X T X 12 (object trajectory)
        # condition: BS X T X D_cond (bbox features + pose conditions)
        # noise_t: BS (timestep)

        # DEBUG: Check input dimensions
        print(f"DEBUG TransformerDiffusionModel: src.shape={src.shape}, condition.shape={condition.shape}")
        print("src mean/std after concat:", src.mean().item(), src.std().item())
        # Concatenate trajectory with conditions
        src = torch.cat((src, condition), dim=-1)
        print(f"DEBUG TransformerDiffusionModel: after concat src.shape={src.shape}")
       
        # Get timestep embedding
        noise_t_embed = self.time_mlp(noise_t)  # BS X d_model 
        if language_embedding is not None:
            noise_t_embed += language_embedding  # BS X d_model 
        noise_t_embed = noise_t_embed[:, None, :]  # BS X 1 X d_model 

        bs = src.shape[0]
        num_steps = src.shape[1] + 1  # Following original CHOIS: sequence length + 1

        if padding_mask is None:
            padding_mask = torch.ones(bs, 1, num_steps).to(src.device).bool()

        # Get position vec for position-wise embedding: [1, 2, 3, ..., T+1]
        pos_vec = torch.arange(num_steps) + 1
        pos_vec = pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1)

        data_input = src.transpose(1, 2)  # BS X D X T 
        feat_pred, _ = self.motion_transformer(data_input, padding_mask, pos_vec, obj_embedding=noise_t_embed)
       
        output = self.linear_out(feat_pred[:, 1:])  # BS X T X 12 - take positions 1 to T (skip position 0)

        return output  # predicted noise/clean trajectory, same size as input

class ObjectTrajectoryDiffusion(nn.Module):
    def __init__(
        self,
        d_feats=12,  # Object trajectory dimension [xyz + 6D rotation]
        d_model=512,
        n_head=8,
        n_dec_layers=6,
        d_k=64,
        d_v=64,
        max_timesteps=101,  # trajectory_length + 1
        timesteps=1000,
        loss_type='l1',
        objective='pred_x0',
        beta_schedule='cosine',
        p2_loss_weight_gamma=0.,
        p2_loss_weight_k=1,
        use_bps=False,
        bps_input_dim=3072,
        bps_hidden_dim=512,
        bps_output_dim=256,
        bps_num_points=1024,
        use_text_embedding=False
    ):
        super().__init__()
        self.use_text_embedding = use_text_embedding
        self.use_bps = use_bps
        self.bps_num_points = bps_num_points
        self.bps_input_dim = bps_input_dim

        # BPS encoder for object shape (following CHOIS)
        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=self.bps_input_dim, out_features=bps_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=bps_hidden_dim, out_features=bps_output_dim),
        )

        # CLIP text encoder (same as chois)
        self.clip_encoder = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
        )

        obj_feats_dim = bps_output_dim 
        
        # d_input_feats for TransformerDiffusionModel is the sum of noisy trajectory and condition dimensions
        d_cond_feats = d_feats  # for pose condition
        if self.use_bps:
            d_cond_feats += obj_feats_dim

        # Total input dimension for the motion transformer
        d_input_feats = d_feats + d_cond_feats

        self.denoise_fn = TransformerDiffusionModel(
            d_input_feats=d_input_feats,
            d_feats=d_feats, 
            d_model=d_model, 
            n_head=n_head, 
            d_k=d_k, 
            d_v=d_v,
            n_dec_layers=n_dec_layers, 
            max_timesteps=max_timesteps
        )
        
        self.objective = objective
        self.seq_len = max_timesteps - 1
        
        # Diffusion schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # Register diffusion buffers
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # P2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def create_sparse_condition_mask(self, seq_len, batch_size, device, history_fraction=0.3):
        """
        创建稀疏条件掩码：历史部分（按history_fraction）+ 最终位姿
        
        Args:
            seq_len: 序列长度
            batch_size: batch大小 
            device: 设备
            history_fraction: 历史部分占总序列长度的比例
            
        Returns:
            cond_mask: BS X T X 12, 0表示已知条件，1表示需要生成
        """
        # Initialize mask - all positions need to be generated (1)
        cond_mask = torch.ones(batch_size, seq_len, 12, device=device)
        
        # 1. 计算历史部分长度
        history_length = max(1, int(seq_len * history_fraction))
        
        # 2. 历史部分的完整位姿作为条件
        cond_mask[:, :history_length, :] = 0  # Complete history poses
        
        # 3. 最终位姿作为条件 
        if seq_len > history_length:
            cond_mask[:, -1, :] = 0  # Final pose (xyz + rotation)
        
        return cond_mask

    def create_condition_by_strategy(self, x_start_12d, attention_mask, seq_len, batch_size, device, 
                                   strategy='history_fraction', history_fraction=0.3, waypoint_interval=30):
        """
        根据不同策略创建条件
        
        Args:
            x_start_12d: BS X T X 12, 完整轨迹
            attention_mask: BS X T, a mask where 1 indicates a valid frame
            seq_len: 序列长度 (padded)
            batch_size: batch大小
            device: 设备
            strategy: 条件策略
                - 'full_trajectory': 直接使用完整轨迹作为条件
                - 'history_fraction': 根据fraction给定一定比例的input部分 + endpose
                - 'chois_original': 和原始chois一样，给定startpose + 每隔一定区间的xy + endpose的xyz
            history_fraction: 历史部分比例
            waypoint_interval: 路径点间隔
            
        Returns:
            x_cond: BS X T X 12, 条件轨迹
        """
        if strategy == 'full_trajectory':
            # 直接使用完整轨迹作为条件
            return x_start_12d.clone()

        x_cond = torch.zeros_like(x_start_12d)
        actual_lengths = attention_mask.sum(dim=1).int()

        for i in range(batch_size):
            length = actual_lengths[i].item()
            if length == 0:
                continue

            if strategy == 'history_fraction':
                # 根据fraction给定一定比例的input部分 + endpose
                history_length = max(1, int(length * history_fraction))
                
                # 历史部分的完整位姿
                x_cond[i, :history_length, :] = x_start_12d[i, :history_length, :].clone()
                
                # 最终位姿
                if length > history_length:
                    x_cond[i, length - 1, :] = x_start_12d[i, length - 1, :].clone()
                    
            elif strategy == 'chois_original':
                # 和原始chois一样，给定startpose + 每隔一定区间的xy + endpose的xyz
                
                # 起始帧的完整位姿
                x_cond[i, 0, :] = x_start_12d[i, 0, :].clone()
                
                # 最终帧的位置（xyz，不包括旋转）
                if length > 1:
                    x_cond[i, length - 1, :3] = x_start_12d[i, length - 1, :3].clone()
                
                # 每隔waypoint_interval帧设置xy路径点
                waypoint_frames = list(range(waypoint_interval, length - 1, waypoint_interval))
                for frame_idx in waypoint_frames:
                    x_cond[i, frame_idx, :2] = x_start_12d[i, frame_idx, :2].clone()  # 只给xy
            
            else:
                # This case should ideally not be reached if strategy is validated before
                raise ValueError(f"Unknown conditioning strategy for batch processing: {strategy}")
                
        return x_cond

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, x_cond, language_embedding=None, padding_mask=None, clip_denoised=True):
        model_output = self.denoise_fn(x, t, x_cond, language_embedding, padding_mask)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # if clip_denoised:
        #     x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, x_cond, language_embedding=None, padding_mask=None, clip_denoised=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, x_cond=x_cond, 
                                                                language_embedding=language_embedding,
                                                                padding_mask=padding_mask, 
                                                                clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x_cond_override=None, language_embedding=None, padding_mask=None):
        """DEBUG: 简化的采样循环，支持自定义条件"""
        device = self.betas.device
        b = shape[0]
        seq_len = shape[1]
        x = torch.randn(shape, device=device)

        # DEBUG: Use simplified condition
        if x_cond_override is not None:
            x_cond = x_cond_override  # Use provided condition (BS X T X 12)
        else:
            # Fallback: use zero condition for sampling
            x_cond = torch.zeros(b, seq_len, 12, device=device)  # Zero condition
            print("DEBUG: Using zero condition for sampling")

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), 
                            x_cond, language_embedding=language_embedding, padding_mask=padding_mask)

        return x

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
        
    @torch.no_grad()
    def sample(self, sequence_length, batch_size=1, language_text=None, bbox_corners=None, device=None, x_cond_override=None):
        """
        DEBUG: 简化的采样接口
        
        Args:
            sequence_length: int, desired trajectory length
            batch_size: int, batch size
            language_text: Optional text prompt (ignored in DEBUG mode)
            bbox_corners: Optional bbox corners (ignored in DEBUG mode)
            device: torch device
            x_cond_override: Optional custom condition (BS X T X 12)
            
        Returns:
            Generated trajectories: BS X T X 12
        """
        if device is None:
            device = next(self.parameters()).device
            
        shape = (batch_size, sequence_length, 12)  # d_feats = 12
        
        # Create padding mask (assume full length for sampling)
        padding_mask = torch.ones(batch_size, 1, sequence_length + 1, device=device)
        
        # DEBUG: Ignore language and BPS conditions for simplicity
        language_embedding = None
        print("DEBUG: Language and BPS conditions are ignored in DEBUG mode")
        
        # Sample using simplified logic
        samples = self.p_sample_loop(shape, x_cond_override=x_cond_override, 
                                   language_embedding=language_embedding, 
                                   padding_mask=padding_mask)
        
        return samples
    
    def p_losses(self, x_start, t, x_cond, language_embedding=None, noise=None, padding_mask=None):
        """
        按照原始 CHOIS 的逻辑计算损失
        
        Args:
            x_start: BS X T X 12, clean trajectories
            x_cond: BS X T X D_cond, condition features
            t: BS, noise levels
            language_embedding: BS X 512, text features
            padding_mask: BS X 1 X T, sequence mask
            
        Returns:
            model_out: BS X T X 12, model prediction
            loss: scalar, computed loss
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        # Add noise to clean trajectories
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        # Predict using denoising network
        model_out = self.denoise_fn(x, t, x_cond, language_embedding=language_embedding, padding_mask=padding_mask)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # Calculate loss with padding mask

        if padding_mask is not None:
            loss = self.loss_fn(model_out, target, reduction='none') * padding_mask[:, 0, 1:][:, :, None]
        else:
            loss = self.loss_fn(model_out, target, reduction='none')

        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        return model_out, loss.mean()

    def forward(self, batch_data, history_fraction=0.3, conditioning_strategy='history_fraction', waypoint_interval=30):
        """
        前向传播，按照原始 CHOIS 的简化逻辑
        
        Args:
            batch_data: dict, 
                - poses: BS X trajectory_length X 9 (will be converted to 12D internally)
                - attention_mask: BS X trajectory_length
                - object_category: List of category strings (optional)
                - text_features: BS X 512 (optional)
                - bbox_corners: BS X trajectory_length X 8 X 3 (object bbox corners, optional)
            history_fraction: 历史部分占总序列长度的比例
            conditioning_strategy: 条件策略 ('full_trajectory', 'history_fraction', 'chois_original')
            waypoint_interval: 路径点间隔(用于chois_original策略)
            
        Returns:
            model_out: BS X T X 12, model prediction
            loss: scalar, computed loss
        """
        # Extract data from batch
        bs, seq_len = batch_data['poses'].shape[:2]
        device = batch_data['poses'].device
        #---1.0 init random t
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (bs,), device=device).long()
        print("noise level: ", t)
        
        #---1.1 prepare x_start
        x_start = batch_data['poses']  # BS X T X 9 -> need to convert to 12D
        attention_mask = batch_data['attention_mask']  # BS X T
        
        # Convert 9D to 12D (from xyz+6d_rot to xyz+9d_rot_matrix)
        # Extract positions and 6D rotations
        positions = x_start[:, :, :3]  # BS X T X 3
        rot_6d = x_start[:, :, 3:9]   # BS X T X 6
        
        # Convert 6D rotation to rotation matrix
        rot_matrices = rotation_6d_to_matrix_torch(rot_6d.view(-1, 6))  # (BS*T) X 3 X 3
        rot_matrices = rot_matrices.view(bs, seq_len, 3, 3)  # BS X T X 3 X 3
        rot_matrices_flat = rot_matrices.view(bs, seq_len, 9)  # BS X T X 9
        
        # Combine to 12D representation
        x_start_12d = torch.cat([positions, rot_matrices_flat], dim=-1)  # BS X T X 12
        
        #---1.3 prepare x_cond (pose)
        # Create pose condition using the specified strategy
        x_pose_cond = self.create_condition_by_strategy(
            x_start_12d, attention_mask, seq_len, bs, device, 
            strategy=conditioning_strategy, 
            history_fraction=history_fraction, 
            waypoint_interval=waypoint_interval
        )  # BS X T X 12

        if self.use_bps:
            # BPS features (object shape conditioning)
            if 'bbox_corners' in batch_data and batch_data['bbox_corners'] is not None:
                # Compute BPS from real bbox corners
                ori_x_cond = compute_bps_from_bbox_corners(batch_data['bbox_corners'], num_points=self.bps_num_points)
            else:
                # Fallback to dummy BPS representation
                print("Warning: No bbox_corners provided, using dummy BPS representation")
                ori_x_cond = torch.randn(bs, 1, self.bps_input_dim, device=device)
        
            bps_feats = self.bps_encoder(ori_x_cond) # BS X 1 X D_bps
            bps_feats = bps_feats.repeat(1, seq_len, 1) # BS X T X D_bps

            x_cond = torch.cat((x_pose_cond, bps_feats), dim=-1)
        else:
            x_cond = x_pose_cond
            
        # Create padding mask from attention mask
        padding_mask = attention_mask.unsqueeze(1)  # BS X 1 X T
        # Add timestep dimension for compatibility with +1 design
        extended_mask = torch.ones(bs, 1, seq_len + 1, device=device)
        extended_mask[:, :, 1:] = padding_mask
        padding_mask = extended_mask
        
        #---1.4 prepare lang embedding
        # Process text if available
        language_embedding = None
        if self.use_text_embedding:
            if 'text_features' in batch_data and batch_data['text_features'] is not None:
                language_embedding = self.clip_encoder(batch_data['text_features'])
        
        # Calculate loss using simplified CHOIS logic
        model_out, loss = self.p_losses(x_start_12d, t, x_cond, 
                                       language_embedding=language_embedding, padding_mask=padding_mask)
        
        return model_out, loss


    def convert_12d_to_9d(self, trajectory_12d):
        """
        将 12D 轨迹转换回 9D 格式 (xyz + 6D rotation)
        
        Args:
            trajectory_12d: BS X T X 12
            
        Returns:
            trajectory_9d: BS X T X 9
        """
        bs, seq_len = trajectory_12d.shape[:2]
        
        # Extract positions and rotation matrices
        positions = trajectory_12d[:, :, :3]  # BS X T X 3
        rot_matrices_flat = trajectory_12d[:, :, 3:12]  # BS X T X 9
        rot_matrices = rot_matrices_flat.view(bs, seq_len, 3, 3)  # BS X T X 3 X 3
        
        # Convert rotation matrices to 6D representation
        rot_6d = rotation_matrix_to_6d_torch(rot_matrices.view(-1, 3, 3))  # (BS*T) X 6
        rot_6d = rot_6d.view(bs, seq_len, 6)  # BS X T X 6
        
        # Combine to 9D representation
        trajectory_9d = torch.cat([positions, rot_6d], dim=-1)  # BS X T X 9
        
        return trajectory_9d 

def compute_bps_from_bbox_corners(bbox_corners, num_points=1024):
    """
    Compute BPS (Basis Point Set) representation from bounding box corners.
    
    Args:
        bbox_corners: Tensor of shape [B, T, 8, 3] containing 8 corners of each bbox
        num_points: Number of points to sample for BPS representation
        
    Returns:
        BPS representation: Tensor of shape [B, 1, num_points*3]
    """
    batch_size, seq_len, num_corners, _ = bbox_corners.shape
    device = bbox_corners.device
    
    # Use the first frame's bbox corners for BPS representation
    # This assumes the object shape doesn't change significantly during the trajectory
    first_frame_corners = bbox_corners[:, 0, :, :]  # [B, 8, 3]
    
    # Create BPS by sampling points from the bbox
    bps_points = []
    
    for b in range(batch_size):
        corners = first_frame_corners[b]  # [8, 3]
        
        # Method 1: Use the 8 corners directly + sample additional points
        points = [corners]  # Start with the 8 corners
        
        # Method 2: Sample points on the faces of the bbox
        if num_points > 8:
            remaining_points = num_points - 8
            
            # Sample points on bbox faces and inside the bbox
            for _ in range(remaining_points):
                # Sample random barycentric coordinates for the bbox
                # Sample a random point inside the bbox using min/max bounds
                min_coords = torch.min(corners, dim=0)[0]  # [3]
                max_coords = torch.max(corners, dim=0)[0]  # [3]
                
                # Random point inside bbox
                random_point = min_coords + torch.rand(3, device=device) * (max_coords - min_coords)
                points.append(random_point.unsqueeze(0))
        
        # Concatenate all points
        bbox_points = torch.cat(points, dim=0)  # [num_points, 3]
        
        # If we have more points than needed, randomly sample
        if bbox_points.shape[0] > num_points:
            indices = torch.randperm(bbox_points.shape[0], device=device)[:num_points]
            bbox_points = bbox_points[indices]
        # If we have fewer points, repeat some points
        elif bbox_points.shape[0] < num_points:
            repeat_factor = (num_points + bbox_points.shape[0] - 1) // bbox_points.shape[0]
            bbox_points = bbox_points.repeat(repeat_factor, 1)[:num_points]
        
        bps_points.append(bbox_points)
    
    # Stack all bbox points: [B, num_points, 3]
    bps_tensor = torch.stack(bps_points, dim=0)
    
    # Reshape to [B, 1, num_points*3] to match expected BPS format
    bps_flat = bps_tensor.view(batch_size, 1, -1)
    
    return bps_flat 