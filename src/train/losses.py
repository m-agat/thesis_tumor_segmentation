from monai.losses import DiceLoss, GeneralizedDiceLoss
from monai.transforms import AsDiscrete, Activations
import config

# Loss functions
def get_loss_function(binary_segmentation=False):
    if binary_segmentation:
        return DiceLoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=True
        )
    else:
        return GeneralizedDiceLoss(
            include_background=True,
            to_onehot_y=False,
            softmax=True,
            w_type="square"
        )
        

def get_activation_function(binary_segmentation=False):
    if binary_segmentation:
        post_activation = Activations(sigmoid=True)
        post_pred = AsDiscrete(argmax=False, threshold=0.5)
        return post_activation, post_pred
    else:
        post_activation = Activations(softmax=True)
        post_pred = AsDiscrete(argmax=False, threshold=None) # No threshold to keep probability maps
        return post_activation, post_pred