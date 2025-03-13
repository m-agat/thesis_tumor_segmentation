from monai.networks.nets import SwinUNETR, SegResNet, VNet, AttentionUnet
import config.config as config

swinunetr_model = SwinUNETR(
    img_size=config.roi,
    in_channels=4,
    out_channels=4,  
    feature_size=48,
    drop_rate=0.2,
    attn_drop_rate=0.2,
    dropout_path_rate=0.2,
    use_checkpoint=True,
).to(config.device)

segresnet_model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=4,  
    dropout_prob=0.2,
).to(config.device)

attunet_model = AttentionUnet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,  
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 1),
    dropout=0.2
).to(config.device)

vnet_model = VNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,  
    dropout_prob_down=0.2,
    dropout_prob_up=(0.2, 0.2),
    dropout_dim=3,
    act=("elu", {"inplace": True}),
).to(config.device)


models_dict = {
    "swinunetr_model.pt": lambda: SwinUNETR(img_size=config.roi, in_channels=4, out_channels=4, feature_size=48, drop_rate=0.2, attn_drop_rate=0.2, dropout_path_rate=0.2, use_checkpoint=True),
    "segresnet_model.pt": lambda: SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16, in_channels=4, out_channels=4, dropout_prob=0.2),
    "attunet_model.pt": lambda: AttentionUnet(spatial_dims=3, in_channels=4, out_channels=4, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 1), dropout=0.2),
    "vnet_model.pt": lambda: VNet(spatial_dims=3, in_channels=4, out_channels=4, dropout_prob_down=0.2, dropout_prob_up=(0.2, 0.2), dropout_dim=3, act=("elu", {"inplace": True})),
}

final_models_dict = {
    "swinunetr_model.pt": swinunetr_model,
    "segresnet_model.pt": segresnet_model,
    "attunet_model.pt": attunet_model,
    "vnet_model.pt": vnet_model,
}


# Function to retrieve model name
def get_model_name(models_dict, model_instance):
    for key, value in models_dict.items():
        if value == model_instance:
            return key
    return None