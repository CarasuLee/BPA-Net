import torch
import torch.nn as nn
import sys
import os
import yaml

# Add VMamba path
VMAMBA_PATH = 'VMamba'
if VMAMBA_PATH not in sys.path:
    sys.path.append(VMAMBA_PATH)

try:
    from classification.models.vmamba import VSSM
    from classification.config import _C as _C_VMAMBA 
except ImportError as e:
    print(f"Warning: Could not import VMamba: {e}")

class VMambaBackbone(nn.Module):

    def __init__(self, config_path: str, checkpoint_path: str | None = None):
        super().__init__()
        
        cfg = _C_VMAMBA.clone()
        
        def update_from_file(cfg, file):
            with open(file, 'r') as f:
                y = yaml.load(f, Loader=yaml.FullLoader)
            if 'BASE' in y:
                for base in y['BASE']:
                    if base:
                         join_path = os.path.join(os.path.dirname(file), base)
                         update_from_file(cfg, join_path)
            cfg.merge_from_file(file)
            
        update_from_file(cfg, config_path)
        cfg.freeze()

        self.model = VSSM(
            patch_size=cfg.MODEL.VSSM.PATCH_SIZE, 
            in_chans=cfg.MODEL.VSSM.IN_CHANS, 
            num_classes=cfg.MODEL.NUM_CLASSES, 
            depths=cfg.MODEL.VSSM.DEPTHS, 
            dims=cfg.MODEL.VSSM.EMBED_DIM, 
            ssm_d_state=cfg.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=cfg.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=cfg.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if cfg.MODEL.VSSM.SSM_DT_RANK == "auto" else int(cfg.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=cfg.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=cfg.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=cfg.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=cfg.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=cfg.MODEL.VSSM.SSM_INIT,
            forward_type=cfg.MODEL.VSSM.SSM_FORWARDTYPE,
            mlp_ratio=cfg.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=cfg.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=cfg.MODEL.VSSM.MLP_DROP_RATE,
            drop_path_rate=cfg.MODEL.DROP_PATH_RATE,
            patch_norm=cfg.MODEL.VSSM.PATCH_NORM,
            norm_layer=cfg.MODEL.VSSM.NORM_LAYER,
            downsample_version=cfg.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=cfg.MODEL.VSSM.PATCHEMBED,
            gmlp=cfg.MODEL.VSSM.GMLP,
            use_checkpoint=cfg.TRAIN.USE_CHECKPOINT,
            posembed=cfg.MODEL.VSSM.POSEMBED,
            imgsize=cfg.DATA.IMG_SIZE,
        )

        if checkpoint_path is not None:
            print(f"[VMambaBackbone] Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v
                
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"[VMambaBackbone] Checkpoint loaded: {msg}")

        if hasattr(self.model, 'classifier') and isinstance(self.model.classifier, nn.Sequential):
            self.model.classifier = self.model.classifier[0]
            
    def forward(self, images: torch.Tensor):
        output = self.model(images)
        if isinstance(output, tuple):
            x = output[0]
        else:
            x = output
        
        # Expected: (B, C, H, W) -> (B, 1024, 7, 7)
        
        # Flatten spatial dims and transpose to (B, N, C)
        x = x.flatten(2).transpose(1, 2) 
        # Expected: (B, 49, 1024)

        B, N, C = x.shape
        mask = torch.zeros((B, N), dtype=torch.bool, device=x.device)
        return x, mask

def build_vmamba_backbone():
    """Factory to build VMamba backbone for finetuning."""
    config_path = 'VMamba/classification/configs/vssm/vmambav2_base_224.yaml'
    checkpoint_path = 'VMamba/vssm_base_0229_ckpt_epoch_237.pth'
    
    print(f"Building VMamba Encoder...")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    
    backbone = VMambaBackbone(config_path=config_path, checkpoint_path=checkpoint_path)
    return backbone
