from monai.losses import DiceLoss, GeneralizedDiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Activations
from monai.utils.enums import MetricReduction

import config

# Loss functions
def get_loss_function(use_global):
    if use_global:
        return GeneralizedDiceLoss(
            include_background=True,
            to_onehot_y=False,
            softmax=True,
            w_type="square"
        )
    else:
        return DiceLoss(
            include_background=True,
            to_onehot_y=False,
            softmax=True
        )

# Use softmax activation for probability maps
post_softmax = Activations(softmax=True)
post_pred = AsDiscrete(argmax=False, threshold=None)  # No threshold to keep probability maps

# Metrics
dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
