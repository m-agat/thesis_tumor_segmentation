import random
import sys 
import json 
import os 
import itertools

sys.path.append("../")

# Custom
from cross_validation import cross_validate_trainer
import config.config as config
import models.models as models
from utils.utils import convert_to_serializable

# MONAI
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.losses import GeneralizedDiceFocalLoss
from monai.transforms import AsDiscrete, Activations
                  
import torch
# Enable cuDNN benchmark for optimized performance
torch.backends.cudnn.benchmark = True 

torch.manual_seed(42)     

# Loss 
loss_func = GeneralizedDiceFocalLoss(
    include_background=False, # We focus on subregions, not background
    to_onehot_y=False, # One-hot encoded in the transformations
    sigmoid=False, # Use softmax for multi-class segmentation
    softmax=True, # Multi-class softmax output
    w_type="square"
)

# Dice score
dice_acc = DiceMetric(
    include_background=False, 
    reduction=MetricReduction.MEAN_BATCH, # Compute average Dice for each batch
    get_not_nans=True,
)

# Scheduler
def scheduler_func(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

# Post-processing transforms
post_activation = Activations(softmax=True) # Softmax for multi-class output
post_pred = AsDiscrete(argmax=True, to_onehot=4) # get the class with the highest prob for each channel

# Hyperparmeter tuning 
hyperparameter_space = {
    "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3],
    "optimizer": ["AdamW", "SGD"],
    "weight_decay": [1e-5, 1e-4],
}

configs = random.sample(list(itertools.product(
    hyperparameter_space["learning_rate"],
    hyperparameter_space["optimizer"],
    hyperparameter_space["weight_decay"]
)), 10)

# Store results for each configuration
tuning_results = []

for idx, (lr, opt, wd) in enumerate(configs):
    print(f"\nTesting Configuration {idx + 1}/{len(configs)}")
    print(f"Learning Rate: {lr}, Optimizer: {opt}, Weight Decay: {wd}")

    # Define the optimizer function for this configuration
    def optimizer_func(params):
        if opt == "AdamW":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        elif opt == "SGD":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        
    # Perform cross-validation for this configuration
    fold_results, avg_metrics = cross_validate_trainer(
        model_class=lambda: models.models_dict[f"{config.model_name}_model.pt"],
        optimizer_func=optimizer_func,
        loss_func=loss_func,
        acc_func=dice_acc,
        scheduler_func=scheduler_func,
        num_folds=config.num_folds,
        post_activation=post_activation,
        post_pred=post_pred
    )

    # Save the results
    tuning_results.append({
        "learning_rate": lr,
        "optimizer": opt,
        "weight_decay": wd,
        "avg_dice": avg_metrics.get("avg_dice", None),
        "avg_hd95": avg_metrics.get("avg_hd95", None),
        "avg_sensitivity": avg_metrics.get("avg_sensitivity", None),
        "avg_specificity": avg_metrics.get("avg_specificity", None),
    })

# Sort results by the average Dice score
tuning_results.sort(key=lambda x: x["avg_dice"], reverse=True)
tuning_results = convert_to_serializable(tuning_results)

# Save tuning results to JSON
tuning_results_path = os.path.join(config.output_dir, "hyperparameter_tuning_results.json")
with open(tuning_results_path, "w") as f:
    json.dump(tuning_results, f, indent=4)

print(f"Hyperparameter tuning results saved to {tuning_results_path}")
print("Best Configuration:")
print(tuning_results[0])

# serializable_fold_results = convert_to_serializable(fold_results)
# cv_results_path = os.path.join(config.output_dir, f"{config.model_name}_cv_results.json")
# with open(cv_results_path, "w") as f:
#     json.dump(serializable_fold_results, f, indent=4)

# print(f"Cross-validation results saved to {cv_results_path}")

# tensorboard --logdir=./outputs/runs/experiment_1