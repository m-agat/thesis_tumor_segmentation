import torch
from monai.networks.nets import SwinUNETR, SegResNet, VNet, AttentionUnet
import config

# Initialize models
swinunetr_model = SwinUNETR(
    img_size=config.global_roi,
    in_channels=4,
    out_channels=3,
    feature_size=32,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    dropout_path_rate=0.1,
    use_checkpoint=True,
).to(config.device)

segresnet_model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
).to(config.device)

attunet_model = AttentionUnet(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    dropout=0.1
).to(config.device)

vnet_model = VNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    act=("elu", {"inplace": True}),
).to(config.device)

models_dict = {
    "swinunetr.pt": swinunetr_model,
    "segresnet.pt": segresnet_model,
    "attunet.pt": attunet_model,
    "vnet.pt": vnet_model
}

def get_model_name(models_dict, model_instance):
    for key, value in models_dict.items():
        if value == model_instance:
            return key
    return None
