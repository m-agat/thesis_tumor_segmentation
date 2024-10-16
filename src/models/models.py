from monai.networks.nets import SwinUNETR, SegResNet, VNet, AttentionUnet
import config.config as config

# Models for Stage 1 (Binary Segmentation: WT vs Background)
swinunetr_model_stage1 = SwinUNETR(
    img_size = config.global_roi,
    in_channels=4,
    out_channels=2,  # Binary segmentation (WT vs Background)
    feature_size=48,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    dropout_path_rate=0.1,
    use_checkpoint=True,
).to(config.device)

segresnet_model_stage1 = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=2,  # Binary segmentation
    dropout_prob=0.2,
).to(config.device)

attunet_model_stage1 = AttentionUnet(
    spatial_dims=3,
    in_channels=4,
    out_channels=2,  # Binary segmentation
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    dropout=0.1
).to(config.device)

vnet_model_stage1 = VNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=2,  # Binary segmentation
    act=("elu", {"inplace": True}),
).to(config.device)


# Models for Stage 2 (Multi-class Segmentation: WT, TC, ET, Background)
swinunetr_model_stage2 = SwinUNETR(
    img_size=config.global_roi,
    in_channels=4,
    out_channels=4,  # Multi-class segmentation (WT, TC, ET, Background)
    feature_size=48,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    dropout_path_rate=0.1,
    use_checkpoint=True,
).to(config.device)

segresnet_model_stage2 = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=4,  # Multi-class segmentation
    dropout_prob=0.2,
).to(config.device)

attunet_model_stage2 = AttentionUnet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,  # Multi-class segmentation
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    dropout=0.1
).to(config.device)

vnet_model_stage2 = VNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,  # Multi-class segmentation
    act=("elu", {"inplace": True}),
).to(config.device)


# Store models in dictionaries for easier access
models_stage1 = {
    "swinunetr_stage1.pt": swinunetr_model_stage1,
    "segresnet_stage1.pt": segresnet_model_stage1,
    "attunet_stage1.pt": attunet_model_stage1,
    "vnet_stage1.pt": vnet_model_stage1,
}

models_stage2 = {
    "swinunetr_stage2.pt": swinunetr_model_stage2,
    "segresnet_stage2.pt": segresnet_model_stage2,
    "attunet_stage2.pt": attunet_model_stage2,
    "vnet_stage2.pt": vnet_model_stage2,
}

# Function to retrieve model name
def get_model_name(models_dict, model_instance):
    for key, value in models_dict.items():
        if value == model_instance:
            return key
    return None