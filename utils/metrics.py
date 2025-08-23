# utils/metrics.py

import numpy as np
from scipy.stats import pearsonr, rankdata
from scipy.special import rel_entr
from scipy.spatial.distance import pdist, cdist
import warnings
from typing import Set, Dict, Optional, List

def _find_non_finite_indices(arr: np.ndarray) -> np.ndarray:
    """Helper function to find the indices of non-finite values (nan, inf)."""
    # For multi-dimensional arrays, returns the flattened indices.
    return np.where(~np.isfinite(arr.flatten()))[0]

def _find_negative_indices(arr: np.ndarray) -> np.ndarray:
    """Helper function to find the indices of negative values."""
    return np.where(arr.flatten() < 0)[0]

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mean Squared Error.
    Adds precise checks for input data and clearly distinguishes between ground truth and prediction.
    All calculation results are rounded to four decimal places.
    """
    # Check 1: Are the input arrays empty?
    if y_true.size == 0:
        warnings.warn("Ground truth (`y_true`) array is empty, returning nan.", UserWarning)
        return np.nan
    if y_pred.size == 0:
        warnings.warn("Prediction (`y_pred`) array is empty, returning nan.", UserWarning)
        return np.nan

    # Check 2: Do the input arrays contain nan or inf, and report indices.
    non_finite_true = _find_non_finite_indices(y_true)
    if non_finite_true.size > 0:
        warnings.warn(f"Ground truth (`y_true`) contains nan or infinity at the following indices: {non_finite_true}. This may result in a nan MSE.", UserWarning)

    non_finite_pred = _find_non_finite_indices(y_pred)
    if non_finite_pred.size > 0:
        warnings.warn(f"Prediction (`y_pred`) contains nan or infinity at the following indices: {non_finite_pred}. This may result in a nan MSE.", UserWarning)
        
    # Check 3: Ensure dimensions match.
    if y_true.shape != y_pred.shape:
        warnings.warn(f"Input array shapes do not match: ground truth shape={y_true.shape}, prediction shape={y_pred.shape}.", UserWarning)
        
    mse = np.mean((y_true - y_pred) ** 2)
    return np.round(mse, 4)

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Error.
    This function relies on compute_mse, thus indirectly inheriting its input checks.
    All calculation results are rounded to four decimal places.
    """
    mse = compute_mse(y_true, y_pred)
    if np.isnan(mse):
        # If mse is already nan, return directly to avoid redundant warnings.
        return np.nan
        
    if mse < 0:
        warnings.warn(f"Calculated MSE is negative ({mse}), RMSE result will be nan.", UserWarning)
    
    rmse = np.sqrt(mse)
    return np.round(rmse, 4)

def compute_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Pearson correlation coefficient.
    Returns the r-value. Adds detailed input checks and clearly distinguishes between ground truth and prediction.
    All calculation results are rounded to four decimal places.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Check 1: Ensure dimensions match.
    if y_true_flat.shape != y_pred_flat.shape:
        warnings.warn(f"Flattened array shapes do not match: ground truth shape={y_true_flat.shape}, prediction shape={y_pred_flat.shape}. Returning nan.", UserWarning)
        return np.nan

    # Check 2: Do the input arrays contain nan or inf, and report indices.
    non_finite_true = _find_non_finite_indices(y_true_flat)
    if non_finite_true.size > 0:
        warnings.warn(f"Ground truth (`y_true`) (flattened) contains nan or infinity at the following indices: {non_finite_true}. Pearson correlation will be nan.", UserWarning)
        return np.nan
    
    non_finite_pred = _find_non_finite_indices(y_pred_flat)
    if non_finite_pred.size > 0:
        warnings.warn(f"Prediction (`y_pred`) (flattened) contains nan or infinity at the following indices: {non_finite_pred}. Pearson correlation will be nan.", UserWarning)
        return np.nan

    # Check 3: Is the input array length sufficient?
    if len(y_true_flat) < 2:
        warnings.warn("Input array has fewer than 2 elements, cannot compute Pearson correlation. Returning nan.", UserWarning)
        return np.nan
        
    # Check 4: Is the input array constant (standard deviation is 0)?
    if len(np.unique(y_true_flat)) == 1:
        warnings.warn(f"Ground truth (`y_true`) is a constant array with all values equal to {y_true_flat[0]}. Its standard deviation is 0, Pearson correlation is undefined. Returning nan.", UserWarning)
        return np.nan
    if len(np.unique(y_pred_flat)) == 1:
        warnings.warn(f"Prediction (`y_pred`) is a constant array with all values equal to {y_pred_flat[0]}. Its standard deviation is 0, Pearson correlation is undefined. Returning nan.", UserWarning)
        return np.nan

    try:
        r, _ = pearsonr(y_true_flat, y_pred_flat)
    except ValueError as e:
        warnings.warn(f"scipy.stats.pearsonr raised an exception: {e}. Returning nan.", UserWarning)
        return np.nan
        
    return np.round(r, 4)

def compute_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Computes the KL Divergence D_{KL}(P || Q).
    P (ground truth) and Q (prediction) are treated as raw vectors. This function
    handles negative values by clipping them to 0 and then normalizes the vectors
    to form probability distributions before comparison.
    All calculation results are rounded to four decimal places.
    """
    # Check 1: Are the input arrays empty?
    if p.size == 0:
        warnings.warn("Input array P (`p`) is empty, returning nan.", UserWarning)
        return np.nan
    if q.size == 0:
        warnings.warn("Input array Q (`q`) is empty, returning nan.", UserWarning)
        return np.nan

    # Check 2: Do the input arrays contain nan or inf, and report indices.
    non_finite_p = _find_non_finite_indices(p)
    if non_finite_p.size > 0:
        warnings.warn(f"Input array P (`p`) contains nan or infinity at the following indices: {non_finite_p}. KL divergence will be nan.", UserWarning)
        return np.nan
    
    non_finite_q = _find_non_finite_indices(q)
    if non_finite_q.size > 0:
        warnings.warn(f"Input array Q (`q`) contains nan or infinity at the following indices: {non_finite_q}. KL divergence will be nan.", UserWarning)
        return np.nan

    p_flat = p.flatten()
    q_flat = q.flatten()

    # Step 1: Handle negative values by clipping them to 0.
    # This is necessary because KL divergence is defined for probability distributions, which must be non-negative.
    if np.any(p_flat < 0):
        warnings.warn("Input array P (`p`) contains negative values. Clipping them to 0 for KL divergence calculation.", UserWarning)
        p_flat = np.maximum(p_flat, 0)

    if np.any(q_flat < 0):
        warnings.warn("Input array Q (`q`) contains negative values. Clipping them to 0 for KL divergence calculation.", UserWarning)
        q_flat = np.maximum(q_flat, 0)

    # Step 2: Check if sums are zero after clipping.
    sum_p = np.sum(p_flat)
    sum_q = np.sum(q_flat)
    
    if sum_q == 0:
        warnings.warn("Sum of Q (`q`) is 0 after clipping negatives. KL divergence is inf if any element in P (`p`) is non-zero.", UserWarning)
        # If sum_q is 0, rel_entr will return inf for any p_norm > 0, so we can return early.
        return np.inf
    if sum_p == 0:
        warnings.warn("Sum of P (`p`) is 0 after clipping negatives. If sum of Q (`q`) is not 0, KL divergence is 0.", UserWarning)
        # If p is all zeros, rel_entr(0, q) is 0 for all q.
        return 0.0

    # Step 3: Apply smoothing (add epsilon) and normalize to create probability distributions.
    p_norm = p_flat + epsilon
    q_norm = q_flat + epsilon
    
    p_norm /= np.sum(p_norm)
    q_norm /= np.sum(q_norm)
    
    # Step 4: Compute KL divergence
    kl_div = np.sum(rel_entr(p_norm, q_norm))
    
    if np.isnan(kl_div) or np.isinf(kl_div):
        warnings.warn(f"The calculated KL divergence is {kl_div}. Please check the original input data.", UserWarning)

    return np.round(kl_div, 4)

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mean Absolute Error (MAE) between ground truth and predicted expressions.
    This metric captures overall predictive accuracy across all genes.
    All calculation results are rounded to four decimal places.

    Args:
        y_true (np.ndarray): Ground truth expression values (pseudobulk).
        y_pred (np.ndarray): Predicted expression values (pseudobulk).

    Returns:
        float: The calculated Mean Absolute Error.
    """
    # Check 1: Are the input arrays empty?
    if y_true.size == 0:
        warnings.warn("Ground truth (`y_true`) array is empty, returning nan.", UserWarning)
        return np.nan
    if y_pred.size == 0:
        warnings.warn("Prediction (`y_pred`) array is empty, returning nan.", UserWarning)
        return np.nan

    # Check 2: Do the input arrays contain nan or inf.
    if _find_non_finite_indices(y_true).size > 0 or _find_non_finite_indices(y_pred).size > 0:
        warnings.warn("Input arrays contain non-finite values. MAE will be nan.", UserWarning)
        return np.nan

    # Check 3: Ensure dimensions match.
    if y_true.shape != y_pred.shape:
        warnings.warn(f"Input array shapes do not match: ground truth shape={y_true.shape}, prediction shape={y_pred.shape}. Returning nan.", UserWarning)
        return np.nan
        
    mae = np.mean(np.abs(y_true - y_pred))
    return np.round(mae, 4)

def compute_des(true_de_genes: Set[str], pred_de_genes: Set[str], pred_gene_fold_changes: Optional[Dict[str, float]] = None) -> float:
    """
    Computes the Differential Expression Score (DES) for a single perturbation.
    This score evaluates how accurately the model predicts significant differentially expressed genes.

    Args:
        true_de_genes (Set[str]): A set of gene identifiers for the ground truth DE genes.
        pred_de_genes (Set[str]): A set of gene identifiers for the predicted DE genes.
        pred_gene_fold_changes (Optional[Dict[str, float]]): A dictionary mapping predicted gene identifiers
            to their absolute fold changes. This is required only if the number of predicted DE genes is
            greater than the number of true DE genes.

    Returns:
        float: The calculated Differential Expression Score, ranging from 0 to 1.
    """
    n_true = len(true_de_genes)
    n_pred = len(pred_de_genes)

    if n_true == 0:
        warnings.warn("The set of true DE genes is empty, DES is not well-defined. Returning 0.0 if predictions exist, 1.0 if not.", UserWarning)
        return 1.0 if n_pred == 0 else 0.0

    if n_pred <= n_true:
        intersection_size = len(true_de_genes.intersection(pred_de_genes))
        des = intersection_size / n_true
    else:  # n_pred > n_true
        if pred_gene_fold_changes is None:
            warnings.warn("`n_pred > n_true` but `pred_gene_fold_changes` was not provided. Cannot rank genes. Returning nan.", UserWarning)
            return np.nan
        
        # Filter the fold changes dict to only include genes from the predicted set
        pred_de_genes_fc = {g: fc for g, fc in pred_gene_fold_changes.items() if g in pred_de_genes}
        
        # Sort predicted DE genes by the absolute value of their fold change in descending order
        sorted_genes = sorted(pred_de_genes_fc.keys(), key=lambda g: abs(pred_de_genes_fc.get(g, 0)), reverse=True)
        
        # Take the top n_true genes from the sorted list
        top_n_true_pred_genes = set(sorted_genes[:n_true])
        
        intersection_size = len(true_de_genes.intersection(top_n_true_pred_genes))
        des = intersection_size / n_true
        
    return np.round(des, 4)

def compute_pds(y_pred_all: np.ndarray, y_true_all: np.ndarray) -> float:
    """
    Computes the mean Perturbation Discrimination Score (PDS) over all perturbations.
    This score measures the model's ability to distinguish between different perturbations.
    Note: This implementation does not exclude the target gene from the distance calculation,
    as it requires a mapping from perturbation to target gene index not provided here.

    Args:
        y_pred_all (np.ndarray): A (N, G) array of N predicted pseudobulk expression vectors for G genes.
        y_true_all (np.ndarray): A (N, G) array of N ground truth pseudobulk expression vectors for G genes.

    Returns:
        float: The mean Perturbation Discrimination Score.
    """
    # Ensure inputs are 2D arrays for consistent shape handling.
    # This handles the case of a single perturbation (1D array) by reshaping it.
    if y_true_all.ndim == 1:
        y_true_all = y_true_all.reshape(1, -1)
    if y_pred_all.ndim == 1:
        y_pred_all = y_pred_all.reshape(1, -1)

    n_perturbations, n_genes = y_true_all.shape
    
    if y_pred_all.shape != y_true_all.shape:
        warnings.warn(f"Shape mismatch between predictions ({y_pred_all.shape}) and ground truth ({y_true_all.shape}). Returning nan.", UserWarning)
        return np.nan

    if n_perturbations < 2:
        # If there's only one perturbation, its rank will always be 1, so PDS is 1.
        return 1.0

    pds_scores = []
    for i in range(n_perturbations):
        # Calculate Manhattan (L1) distance from the i-th prediction to all true perturbations
        distances = np.sum(np.abs(y_true_all - y_pred_all[i, :]), axis=1)
        
        # Get the rank of the distance to the corresponding true perturbation.
        # 'min' method assigns the minimum rank to tied elements.
        ranks = rankdata(distances, method='min')
        rank_of_true_match = ranks[i]
        
        # Normalize the rank to get the score for this perturbation
        pds_p = 1 - (rank_of_true_match - 1) / (n_perturbations -1) if n_perturbations > 1 else 1.0
        pds_scores.append(pds_p)
        
    mean_pds = np.mean(pds_scores)
    return np.round(mean_pds, 4)

def compute_overall_score(
    des_prediction: float, pds_prediction: float, mae_prediction: float,
    des_baseline: float, pds_baseline: float, mae_baseline: float
) -> float:
    """
    Calculates the final overall score by averaging the scaled DES, PDS, and MAE scores.
    Scaled scores measure the improvement of the prediction over a baseline model.

    Args:
        des_prediction (float): The DES score for the prediction.
        pds_prediction (float): The PDS score for the prediction.
        mae_prediction (float): The MAE score for the prediction.
        des_baseline (float): The baseline DES score.
        pds_baseline (float): The baseline PDS score.
        mae_baseline (float): The baseline MAE score.

    Returns:
        float: The final overall leaderboard score, from 0 to 1.
    """
    # Scale DES
    if 1 - des_baseline == 0:
        des_scaled = 0.0
        warnings.warn("1 - des_baseline is zero. DES_scaled is set to 0.", UserWarning)
    else:
        des_scaled = (des_prediction - des_baseline) / (1 - des_baseline)

    # Scale PDS
    if 1 - pds_baseline == 0:
        pds_scaled = 0.0
        warnings.warn("1 - pds_baseline is zero. PDS_scaled is set to 0.", UserWarning)
    else:
        pds_scaled = (pds_prediction - pds_baseline) / (1 - pds_baseline)

    # Scale MAE
    if mae_baseline == 0:
        if mae_prediction == 0:
            mae_scaled = 1.0 # Perfect match with a perfect baseline
        else:
            mae_scaled = 0.0 # Any error is infinitely worse than a perfect baseline
        warnings.warn("mae_baseline is zero. MAE_scaled is set to 1.0 for a perfect prediction, 0.0 otherwise.", UserWarning)
    else:
        mae_scaled = (mae_baseline - mae_prediction) / mae_baseline

    # Clip negative scores to 0
    des_scaled = max(0, des_scaled)
    pds_scaled = max(0, pds_scaled)
    mae_scaled = max(0, mae_scaled)

    # Final score is the mean of the three scaled scores
    overall_score = np.mean([des_scaled, pds_scaled, mae_scaled])
    
    return np.round(overall_score, 4)

def compute_edistance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Energy Distance (E-distance) between two samples.
    E-distance is a statistical distance between probability distributions.
    Assumes y_true and y_pred are (n_samples, n_features).
    Reference: https://github.com/sanderlab/scPerturb
    """
    # Ensure inputs are 2D
    if y_true.ndim == 1: y_true = y_true.reshape(1, -1)
    if y_pred.ndim == 1: y_pred = y_pred.reshape(1, -1)

    if y_true.shape[1] != y_pred.shape[1]:
        warnings.warn(f"Feature dimensions do not match: y_true has {y_true.shape[1]} features, y_pred has {y_pred.shape[1]}. Returning nan.", UserWarning)
        return np.nan

    n, m = len(y_true), len(y_pred)
    if n == 0 or m == 0:
        warnings.warn("Input arrays must not be empty. Returning nan.", UserWarning)
        return np.nan

    # Calculate pairwise distances
    d_pred_pred = pdist(y_pred, 'euclidean')
    d_true_true = pdist(y_true, 'euclidean')
    d_pred_true = cdist(y_pred, y_true, 'euclidean')

    # Compute terms of the E-distance formula
    term1 = np.sum(d_pred_true) / (n * m)
    term2 = np.sum(d_pred_pred) / (m * m) if m > 1 else 0
    term3 = np.sum(d_true_true) / (n * n) if n > 1 else 0
    
    e_distance = 2 * term1 - term2 - term3
    
    return np.round(e_distance, 4)

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the R-squared (R2) coefficient of determination.
    All calculation results are rounded to four decimal places.
    """
    if y_true.shape != y_pred.shape:
        # 逐维度取最小长度
        common_shape = tuple(min(d1, d2) for d1, d2 in zip(y_true.shape, y_pred.shape))

        # 双向裁剪
        slices = tuple(slice(0, dim) for dim in common_shape)
        y_true = y_true[slices]
        y_pred = y_pred[slices]

        warnings.warn(
            f"输入数组形状不匹配：真实值 {y_true.shape}，预测值 {y_pred.shape}。"
            f"已按共同长度 {common_shape} 双向裁剪。",
            UserWarning
        )
    if y_true.size == 0:
        warnings.warn("Input arrays are empty. Returning nan.", UserWarning)
        return np.nan

    # Flatten arrays for calculation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Calculate total sum of squares
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    if ss_tot == 0:
        warnings.warn("Total sum of squares is 0 (y_true is constant). R2 is not well-defined. Returning nan.", UserWarning)
        return np.nan

    # Calculate residual sum of squares
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    
    r2 = 1 - (ss_res / ss_tot)
    return np.round(r2, 4)

def _rbf_kernel(X, Y, gamma=1.0):
    """Helper function to compute the RBF kernel."""
    dist_sq = cdist(X, Y, 'sqeuclidean')
    return np.exp(-gamma * dist_sq)

def compute_mmd(y_true: np.ndarray, y_pred: np.ndarray, gamma: Optional[float] = None) -> float:
    """
    Computes the Maximum Mean Discrepancy (MMD) using an RBF kernel.
    MMD measures the distance between two distributions in a reproducing kernel Hilbert space.
    Assumes y_true and y_pred are (n_samples, n_features).
    """
    # Ensure inputs are 2D
    if y_true.ndim == 1: y_true = y_true.reshape(1, -1)
    if y_pred.ndim == 1: y_pred = y_pred.reshape(1, -1)
    
    if y_true.shape[1] != y_pred.shape[1]:
        warnings.warn(f"Feature dimensions do not match: y_true has {y_true.shape[1]} features, y_pred has {y_pred.shape[1]}. Returning nan.", UserWarning)
        return np.nan
        
    n, m = len(y_true), len(y_pred)
    if n == 0 or m == 0:
        warnings.warn("Input arrays must not be empty. Returning nan.", UserWarning)
        return np.nan

    # Gamma for RBF kernel is often set to 1 / (2 * sigma^2), where sigma is the median distance.
    if gamma is None:
        dists = cdist(y_true, y_true, 'euclidean')
        # Use a small epsilon to avoid division by zero if all distances are zero
        median_dist = np.median(dists[np.triu_indices_from(dists, k=1)]) + 1e-6
        gamma = 1.0 / (2 * median_dist**2)

    # Compute kernel matrices
    K_true_true = _rbf_kernel(y_true, y_true, gamma)
    K_pred_pred = _rbf_kernel(y_pred, y_pred, gamma)
    K_true_pred = _rbf_kernel(y_true, y_pred, gamma)

    # Compute MMD^2
    mmd2 = np.mean(K_true_true) + np.mean(K_pred_pred) - 2 * np.mean(K_true_pred)
    
    # MMD is the square root of MMD^2. Clip at 0 to avoid numerical issues.
    mmd = np.sqrt(max(0, mmd2))
    
    return np.round(mmd, 4)

def compute_pearson_delta(y_true: np.ndarray, y_pred: np.ndarray, y_control: np.ndarray) -> float:
    """
    Computes the Pearson correlation of the *change* in expression (delta).
    Delta is calculated as post-perturbation minus control expression.
    """
    # Calculate deltas
    delta_true = y_true - y_control
    delta_pred = y_pred - y_control
    
    # Reuse the main pearson function for the actual calculation
    return compute_pearson(delta_true, delta_pred)

def _get_top_k_de_indices(y_true: np.ndarray, y_control: np.ndarray, k: int) -> np.ndarray:
    """Helper to find indices of top k differentially expressed genes."""
    if k <= 0:
        return np.array([], dtype=int)
        
    delta_true = y_true.flatten() - y_control.flatten()
    
    # Get indices that would sort the absolute deltas in descending order
    sorted_indices = np.argsort(np.abs(delta_true))[::-1]
    
    # Return the top k indices
    return sorted_indices[:k]

def compute_pearson_de(y_true: np.ndarray, y_pred: np.ndarray, y_control: np.ndarray, k: int, exclude_targets: Optional[List[int]] = None) -> float:
    """
    Computes Pearson correlation on the top k most differentially expressed (DE) genes.
    DE genes are determined from the true data (y_true vs y_control).
    """
    if k > y_true.size:
        warnings.warn(f"k ({k}) is larger than the number of genes ({y_true.size}). Using all genes.", UserWarning)
        k = y_true.size
        
    de_indices = _get_top_k_de_indices(y_true, y_control, k)

    if exclude_targets:
        de_indices = np.setdiff1d(de_indices, exclude_targets, assume_unique=True)

    if de_indices.size == 0:
        warnings.warn("No DE genes to compute correlation on after filtering. Returning nan.", UserWarning)
        return np.nan

    # Select the top k genes from true and pred vectors
    y_true_de = y_true.flatten()[de_indices]
    y_pred_de = y_pred.flatten()[de_indices]
    
    return compute_pearson(y_true_de, y_pred_de)

def compute_pearson_delta_de(y_true: np.ndarray, y_pred: np.ndarray, y_control: np.ndarray, k: int, exclude_targets: Optional[List[int]] = None) -> float:
    """
    Computes Pearson correlation of deltas on the top k most differentially expressed (DE) genes.
    DE genes are determined from the true data (y_true vs y_control).
    """
    if k > y_true.size:
        warnings.warn(f"k ({k}) is larger than the number of genes ({y_true.size}). Using all genes.", UserWarning)
        k = y_true.size

    de_indices = _get_top_k_de_indices(y_true, y_control, k)

    if exclude_targets:
        de_indices = np.setdiff1d(de_indices, exclude_targets, assume_unique=True)

    if de_indices.size == 0:
        warnings.warn("No DE genes to compute correlation on after filtering. Returning nan.", UserWarning)
        return np.nan

    # Select the top k genes from all vectors
    y_true_de = y_true.flatten()[de_indices]
    y_pred_de = y_pred.flatten()[de_indices]
    y_control_de = y_control.flatten()[de_indices]

    # Calculate deltas for the subset of DE genes
    delta_true_de = y_true_de - y_control_de
    delta_pred_de = y_pred_de - y_control_de
    
    return compute_pearson(delta_true_de, delta_pred_de)