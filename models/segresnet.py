from monai.networks.nets import SegResNet

def get_segresnet(in_channels=4, out_channels=5):
    model = SegResNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels, init_filters=16)
    return model
