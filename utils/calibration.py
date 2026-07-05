import numpy as np

def calibrate_confidence(probabilities_dict, temperature=1.5, max_confidence=99.93):
    """
    Calibrate model probabilities using temperature scaling and a soft cap.
    
    Args:
        probabilities_dict: Dictionary mapping class names to percentage probabilities (0-100).
        temperature: Factor to soften the probabilities. T > 1.0 makes the distribution flatter.
                     T = 1.0 is no change.
        max_confidence: The maximum allowed confidence percentage (e.g. 99.93%).
    
    Returns:
        calibrated_probs: Dictionary mapping class names to calibrated percentage probabilities.
        max_class: The class with the highest calibrated probability.
        max_conf: The maximum calibrated confidence score.
    """
    classes = list(probabilities_dict.keys())
    # Convert percentages to 0-1 range
    probs = np.array([probabilities_dict[c] / 100.0 for c in classes], dtype=np.float64)
    
    # Avoid log(0) and extreme saturation by adding small epsilon
    epsilon = 1e-9
    probs = np.clip(probs, epsilon, 1.0 - epsilon)
    
    # Since we have probabilities, we can approximate logits as log(P)
    # Then apply temperature scaling: logit / T
    logits = np.log(probs)
    scaled_logits = logits / temperature
    
    # Softmax to get new probabilities
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits)) # Subtract max for numerical stability
    calibrated_probs = exp_logits / np.sum(exp_logits)
    
    # Apply a soft cap to prevent perfectly certain 100% predictions
    # If the highest probability exceeds max_confidence / 100.0, we redistribute
    # the excess evenly among other classes (or proportionally).
    max_conf_ratio = max_confidence / 100.0
    
    max_idx = np.argmax(calibrated_probs)
    if calibrated_probs[max_idx] > max_conf_ratio:
        excess = calibrated_probs[max_idx] - max_conf_ratio
        calibrated_probs[max_idx] = max_conf_ratio
        
        # distribute excess to others
        other_indices = [i for i in range(len(calibrated_probs)) if i != max_idx]
        if other_indices:
            redist_amount = excess / len(other_indices)
            for i in other_indices:
                calibrated_probs[i] += redist_amount
                
    # Re-normalize just in case floating point errors crept in
    calibrated_probs = calibrated_probs / np.sum(calibrated_probs)
    
    # Construct return dict with percentages
    ret_dict = {classes[i]: float(calibrated_probs[i] * 100.0) for i in range(len(classes))}
    max_c = classes[np.argmax(calibrated_probs)]
    max_val = float(np.max(calibrated_probs) * 100.0)
    
    return ret_dict, max_c, max_val
