from monai.networks.nets import VNet

def get_vnet(in_channels=4, out_channels=5):
    model = VNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels)
    return model
