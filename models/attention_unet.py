from monai.networks.nets import AttentionUnet

def get_attention_unet(in_channels=4, out_channels=5):
    model = AttentionUnet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels)
    return model
