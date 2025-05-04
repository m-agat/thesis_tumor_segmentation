import numpy as np
from typing import Optional, Tuple
from typing import Dict

def extract_confidence_and_correctness(
    probabilities: np.ndarray,
    gt: np.ndarray,
    subregion_label: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    From a softmax volume and ground truth, return
    flattened arrays of confidences and correctness.

    Args:
        probabilities: np.ndarray, shape (C, H, W, D)
        gt:             np.ndarray, shape (H, W, D), integer labels
        subregion_label: int or None -- if given, only keep voxels where gt == label

    Returns:
        confidences: 1-D float array in [0,1]
        correctness: 1-D float array in {0,1}
    """
    # 1) get per-voxel pred & conf
    preds = np.argmax(probabilities, axis=0)
    confs = np.max(probabilities, axis=0)
    correct = (preds == gt).astype(np.float32)

    # 2) apply mask for a subregion, if requested
    if subregion_label is not None:
        mask = (gt == subregion_label)
        return confs[mask], correct[mask]

    # 3) flatten everything
    return confs.ravel(), correct.ravel()


def compute_ece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    num_bins: int = 10
) -> float:
    """
    Expected Calibration Error.

    Args:
        confidences: 1-D float array of model confidences in [0,1]
        correctness: 1-D float array of 0/1 correctness values
        num_bins:    number of equal-width bins over [0,1]

    Returns:
        ece: float scalar
    """
    n = confidences.size
    if n == 0:
        return 0.0

    # 1) assign each sample to a bin
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    # digitize returns 1..num_bins (so subtract 1 → 0..num_bins-1)
    bin_idx = np.digitize(confidences, bin_edges[:-1], right=False) - 1
    # clip any 1.0 values into last bin
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    # 2) compute counts and sums per bin
    counts    = np.bincount(bin_idx, minlength=num_bins)
    sum_conf  = np.bincount(bin_idx, weights=confidences, minlength=num_bins)
    sum_acc   = np.bincount(bin_idx, weights=correctness, minlength=num_bins)

    # 3) only look at bins with at least one sample
    valid = counts > 0
    avg_conf = sum_conf[valid] / counts[valid]
    avg_acc  = sum_acc[valid] / counts[valid]
    weights  = counts[valid] / n

    # 4) ECE = ∑ (|avg_conf - avg_acc| * bin_weight)
    ece = np.sum(weights * np.abs(avg_conf - avg_acc))
    return float(ece)

def compute_ece_per_subregion(
    probabilities: np.ndarray,
    gt: np.ndarray,
    subregions: Dict[str, int],
    num_bins: int
) -> Dict[str, float]:
    """
    For each named subregion, extract confidences+correctness and compute ECE.
    Returns a dict mapping subregion name → ECE.
    """
    ece_results: Dict[str, float] = {}
    for name, label in subregions.items():
        conf, corr = extract_confidence_and_correctness(
            probabilities, gt, subregion_label=label
        )
        ece_results[name] = compute_ece(conf, corr, num_bins=num_bins)
    return ece_results
