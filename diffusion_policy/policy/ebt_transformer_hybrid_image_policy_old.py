# diffusion_policy/policy/ebt_transformer_hybrid_image_policy.py

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import logging

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.ebt.transformer_for_ebt import TransformerForEBT
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc

logger = logging.getLogger(__name__)


class EBTTransformerHybridImagePolicy(BaseImagePolicy):
    """
    Energy-Based Training (EBT) policy using transformer backbone.
    Uses RoboMimic's vision encoder (same as DiffusionTransformerHybridImagePolicy).
    """
    
    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        # MCMC parameters
        num_inference_steps: int = 10,
        mcmc_step_size: float = 0.01,
        mcmc_step_size_learnable: bool = True,
        langevin_noise_std: float = 0.001,
        langevin_noise_learnable: bool = False,
        # Initialization
        initial_action_std: float = 1.0,
        initial_action_type: str = 'gaussian',
        # Gradient clipping
        clip_grad: bool = True,
        clip_grad_value: float = 1.0,
        # Action clamping
        clamp_actions: bool = True,
        clamp_actions_value: float = 10.0,
        # Transformer parameters
        n_layer: int = 8,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        causal_attn: bool = True,
        energy_head_hidden_dim: int = None,
        # Vision encoder
        crop_shape: Optional[Tuple[int, int]] = (76, 76),
        obs_encoder_group_norm: bool = False,
        eval_fixed_crop: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # Parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            
            type_name = attr.get('type', 'low_dim')
            if type_name == 'rgb':
                obs_config['rgb'].append(key)
            elif type_name == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type_name}")
        
        # Get RoboMimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph'
        )
        
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw
        
        ObsUtils.initialize_obs_utils_with_config(config)
        
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu',
        )
        
        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features
                )
            )
        
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )
        
        obs_feature_dim = obs_encoder.output_shape()[0]
        
        # Build EBT transformer
        self.model = TransformerForEBT(
            action_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=obs_feature_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            energy_head_hidden_dim=energy_head_hidden_dim,
        )
        
        # MCMC parameters (learnable)
        self.alpha = nn.Parameter(
            torch.tensor(mcmc_step_size),
            requires_grad=mcmc_step_size_learnable
        )
        self.langevin_noise_std = nn.Parameter(
            torch.tensor(langevin_noise_std),
            requires_grad=langevin_noise_learnable
        )
        
        self.obs_encoder = obs_encoder
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.num_inference_steps = num_inference_steps
        self.initial_action_std = initial_action_std
        self.initial_action_type = initial_action_type
        self.clip_grad = clip_grad
        self.clip_grad_value = clip_grad_value
        self.clamp_actions = clamp_actions
        self.clamp_actions_value = clamp_actions_value
        
        logger.info(
            f"EBT Policy initialized - horizon: {horizon}, "
            f"n_action_steps: {n_action_steps}, "
            f"n_obs_steps: {n_obs_steps}, "
            f"action_dim: {action_dim}, "
            f"obs_feature_dim: {obs_feature_dim}"
        )

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self,
        learning_rate: float = 1e-4,
        transformer_weight_decay: float = 1e-3,
        obs_encoder_weight_decay: float = 1e-6,
        betas: Tuple[float, float] = (0.9, 0.95),
        mcmc_lr_scale: float = 0.1,
    ):
        """Create optimizer."""
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay
        )
        
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        
        mcmc_params = []
        if self.alpha.requires_grad:
            mcmc_params.append(self.alpha)
        if self.langevin_noise_std.requires_grad:
            mcmc_params.append(self.langevin_noise_std)
        
        if len(mcmc_params) > 0:
            optim_groups.append({
                'params': mcmc_params,
                'weight_decay': 0.0,
                'lr': learning_rate * mcmc_lr_scale
            })
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        logger.info(f"Optimizer created with {len(optim_groups)} param groups")
        return optimizer

    def _corrupt_actions(self, shape, device, dtype):
        """Initialize corrupted actions for MCMC."""
        if self.initial_action_type == 'gaussian':
            return torch.randn(shape, device=device, dtype=dtype) * self.initial_action_std
        elif self.initial_action_type == 'zeros':
            return torch.zeros(shape, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown initial_action_type: {self.initial_action_type}")

    def _mcmc_step(self, actions, obs_features, step_idx, total_steps, 
                   create_graph=False, add_noise=True):
        """Perform one MCMC gradient descent step."""
        
        with torch.enable_grad():  # Enable gradients for MCMC
            actions = actions.detach().requires_grad_(True)
            
            if add_noise and self.langevin_noise_std > 0:
                noise = torch.randn_like(actions) * self.langevin_noise_std
                actions_noisy = actions + noise
            else:
                actions_noisy = actions
            
            energy = self.model(actions=actions_noisy, cond=obs_features)
            total_energy = energy.sum()
            
            grad = torch.autograd.grad(
                outputs=total_energy,
                inputs=actions,
                create_graph=create_graph
            )[0]
            
            if self.clip_grad:
                grad = torch.clamp(grad, -self.clip_grad_value, self.clip_grad_value)
            
            alpha = torch.clamp(self.alpha, min=1e-5, max=1.0)
            actions_new = actions - alpha * grad
            
            if self.clamp_actions:
                actions_new = torch.clamp(
                    actions_new,
                    -self.clamp_actions_value,
                    self.clamp_actions_value
                )
        
        # Only detach if not keeping graph for backprop
        if create_graph:
            return actions_new
        else:
            return actions_new.detach()

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute EBT training loss."""
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        
        # Encode observations
        To = self.n_obs_steps
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        obs_features = nobs_features.reshape(batch_size, To, -1)
        
        # Get ground truth actions
        gt_actions = nactions[:, :self.horizon, :]
        
        # Initialize corrupted actions
        predicted_actions = self._corrupt_actions(
            shape=gt_actions.shape,
            device=gt_actions.device,
            dtype=gt_actions.dtype
        )
        
        # MCMC refinement
        for step in range(self.num_inference_steps):
            create_graph = (step == self.num_inference_steps - 1)
            predicted_actions = self._mcmc_step(
                actions=predicted_actions,
                obs_features=obs_features,
                step_idx=step,
                total_steps=self.num_inference_steps,
                create_graph=create_graph,
                add_noise=True
            )
        
        # Compute loss
        loss = nn.functional.mse_loss(predicted_actions, gt_actions)
        return loss

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict actions using MCMC refinement."""
        
        # Remove past_action if present (not implemented yet)
        obs_dict = {k: v for k, v in obs_dict.items() if k != 'past_action'}
        
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        To = self.n_obs_steps
        
        # Encode observations
        with torch.no_grad():
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            obs_features = nobs_features.reshape(B, To, -1)
        
        # Initialize actions
        predicted_actions = self._corrupt_actions(
            shape=(B, self.horizon, self.action_dim),
            device=obs_features.device,
            dtype=obs_features.dtype
        )
        
        # MCMC refinement
        for step in range(self.num_inference_steps):
            predicted_actions = self._mcmc_step(
                actions=predicted_actions,
                obs_features=obs_features,
                step_idx=step,
                total_steps=self.num_inference_steps,
                create_graph=False,
                add_noise=(self.langevin_noise_std > 0)
            )
        
        # Denormalize
        with torch.no_grad():
            action_pred = self.normalizer['action'].unnormalize(predicted_actions)
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]
        
        return {
            'action': action,
            'action_pred': action_pred
        }