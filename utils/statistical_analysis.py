import numpy as np
import scipy.stats as stats

def chi_square_test(observed, expected=None, alpha=0.05):
    """
    Perform a chi-square test to determine if observed values follow the expected distribution.
    
    Args:
        observed: Array of observed frequencies
        expected: Array of expected frequencies (if None, assumes uniform distribution)
        alpha: Significance level (default: 0.05)
        
    Returns:
        tuple: (chi2_statistic, p_value, is_significant)
    """
    if expected is None:
        # Assume uniform distribution if expected not provided
        expected = np.ones_like(observed) * np.sum(observed) / len(observed)
    
    # Calculate chi-square statistic
    chi2_stat = np.sum(((observed - expected) ** 2) / expected)
    
    # Calculate degrees of freedom
    dof = len(observed) - 1
    
    # Calculate p-value
    p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
    
    # Determine if difference is statistically significant
    is_significant = p_value < alpha
    
    return chi2_stat, p_value, is_significant

def chi_square_attack_lsb(data, sample_size=None, pairs=True):
    """
    Perform a chi-square attack on LSB data to detect steganography.
    
    Args:
        data: The data array to analyze
        sample_size: Number of samples to analyze (None for all)
        pairs: Whether to analyze pairs of values (True) or individual samples (False)
        
    Returns:
        tuple: (chi2_statistic, p_value, is_stego_likely)
    """
    if sample_size is not None and sample_size < len(data):
        # Take a random sample to improve performance
        indices = np.random.choice(len(data), sample_size, replace=False)
        sample_data = data[indices]
    else:
        sample_data = data
    
    if pairs:
        # Analyze pairs of adjacent values (PoVs - Pairs of Values)
        # This is more reliable for LSB steganography detection
        histogram = np.zeros(256, dtype=int)
        
        for i in range(0, 256, 2):
            # Count occurrences of value i and i+1
            count_i = np.sum(sample_data == i)
            count_i_plus_1 = np.sum(sample_data == i+1)
            
            # Store the total for this pair
            histogram[i//2] = count_i + count_i_plus_1
        
        # For each pair (i, i+1), the expected distribution should be roughly equal
        # if LSB steganography is present
        observed = np.zeros(128)
        expected = np.zeros(128)
        
        for i in range(0, 256, 2):
            observed[i//2] = np.sum(sample_data == i)
            expected[i//2] = histogram[i//2] / 2  # Equal distribution between i and i+1
        
    else:
        # Analyze individual LSB values
        lsbs = sample_data & 1
        
        # Count occurrences of 0s and 1s
        observed = np.bincount(lsbs, minlength=2)
        
        # Expected frequencies - should be roughly equal for random LSBs
        expected = np.array([len(lsbs) / 2, len(lsbs) / 2])
    
    # Calculate chi-square statistic
    chi2_stat, p_value, is_significant = chi_square_test(observed, expected)
    
    # If p-value is small, it means the distribution is not uniform,
    # suggesting the original cover medium (natural image/audio)
    # If p-value is large, it suggests uniform distribution, which is
    # characteristic of steganography (LSB replacement)
    is_stego_likely = p_value > 0.05  # High p-value suggests steganography
    
    return chi2_stat, p_value, is_stego_likely

def sample_pair_analysis(data, sample_size=None):
    """
    Perform Sample Pair Analysis (SPA) to detect LSB steganography.
    
    Args:
        data: The data array to analyze
        sample_size: Number of samples to analyze (None for all)
    
    Returns:
        float: Estimated embedding rate (0-1), where 0 is no embedding
                and 1 is full embedding
    """
    if sample_size is not None and sample_size < len(data):
        indices = np.random.choice(len(data), sample_size, replace=False)
        sample_data = data[indices]
    else:
        sample_data = data
    
    # Count pairs with specific properties
    count_z = 0  # Count of "Z" pairs (even, even) or (odd, odd)
    count_w = 0  # Count of "W" pairs (even, odd) or (odd, even)
    
    for i in range(len(sample_data) - 1):
        x = sample_data[i]
        y = sample_data[i + 1]
        
        # Check if they form a Z pair or W pair
        if (x % 2 == 0 and y % 2 == 0) or (x % 2 == 1 and y % 2 == 1):
            count_z += 1
        else:
            count_w += 1
    
    # Calculate the embedding rate estimate
    if count_z + count_w > 0:
        embedding_rate = 0.5 - (count_z - count_w) / (2 * (count_z + count_w))
        # Clamp to [0, 1] range
        embedding_rate = max(0, min(1, embedding_rate))
    else:
        embedding_rate = 0
    
    return embedding_rate

def rs_analysis(data, mask=None, sample_size=None, group_size=4):
    """
    Perform RS (Regular/Singular) Analysis to detect LSB steganography.
    
    Args:
        data: The data array to analyze
        mask: Custom mask for discrimination function (default uses [1,-1,1,-1,...])
        sample_size: Number of groups to analyze (None for all)
        group_size: Size of each group
        
    Returns:
        tuple: (embedding_rate, difference, rm_ratio, sm_ratio, r_m_ratio, s_m_ratio)
    """
    # Flatten data if multi-dimensional
    if len(data.shape) > 1:
        data = data.flatten()
    
    # Default mask if not specified
    if mask is None:
        mask = np.array([1, -1] * (group_size // 2))
    
    # Calculate how many groups we can form
    total_groups = len(data) // group_size
    
    # If sample_size is specified, use that many groups
    if sample_size is not None and sample_size < total_groups:
        indices = np.random.choice(total_groups, sample_size, replace=False)
        groups_to_check = indices
    else:
        groups_to_check = range(total_groups)
    
    # Function to calculate discrimination (smoothness measure)
    def discrimination(group):
        return np.sum(np.abs(np.diff(group)))
    
    # Function to apply LSB flipping according to mask (F1 operation)
    def f(group, mask):
        return np.bitwise_xor(group, mask & 1)
    
    # Function to invert LSBs (F-1 operation)
    def f_neg(group, mask):
        return np.bitwise_xor(group, np.bitwise_xor(group & 1, 1) & mask)
    
    # Count regular and singular groups for each operation
    rm = sm = r_m = s_m = 0
    
    for i in groups_to_check:
        # Extract group
        start_idx = i * group_size
        end_idx = min(start_idx + group_size, len(data))
        
        # Skip incomplete groups
        if end_idx - start_idx < group_size:
            continue
            
        group = data[start_idx:end_idx]
        
        # Calculate discrimination for original group
        d_orig = discrimination(group)
        
        # Apply flipping operations
        group_f = f(group, mask)
        group_f_neg = f_neg(group, mask)
        
        # Calculate discrimination after flipping
        d_f = discrimination(group_f)
        d_f_neg = discrimination(group_f_neg)
        
        # Classify groups
        if d_f > d_orig:
            rm += 1  # Regular for positive flipping
        elif d_f < d_orig:
            sm += 1  # Singular for positive flipping
            
        if d_f_neg > d_orig:
            r_m += 1  # Regular for negative flipping
        elif d_f_neg < d_orig:
            s_m += 1  # Singular for negative flipping
    
    # Total number of groups analyzed
    total_analyzed = len(groups_to_check)
    
    # Calculate proportions
    rm_ratio = rm / total_analyzed if total_analyzed > 0 else 0
    sm_ratio = sm / total_analyzed if total_analyzed > 0 else 0
    r_m_ratio = r_m / total_analyzed if total_analyzed > 0 else 0
    s_m_ratio = s_m / total_analyzed if total_analyzed > 0 else 0
    
    # In clean images, rm ≈ r_m and sm ≈ s_m
    # As embedding rate increases, rm and sm approach each other
    
    # Estimate embedding rate based on the difference in the proportions
    d0 = abs(rm_ratio - r_m_ratio)
    d1 = abs(sm_ratio - s_m_ratio)
    
    # Combined difference for more robust estimation
    diff = (d0 + d1) / 2
    
    # Theoretical maximum difference is 0.5 for full embedding
    # Normalize to get an estimate between 0 and 1
    embedding_rate = min(1.0, diff * 2)
    
    return embedding_rate, diff, rm_ratio, sm_ratio, r_m_ratio, s_m_ratio

def perform_detailed_analysis(data, sample_size=10000):
    """
    Perform a comprehensive steganalysis using multiple methods.
    
    Args:
        data: The data array to analyze
        sample_size: Number of samples to analyze
        
    Returns:
        dict: Dictionary of analysis results and metrics
    """
    results = {}
    
    # Flatten the data if needed
    if len(data.shape) > 1:
        flat_data = data.flatten()
    else:
        flat_data = data
        
    # Take a sample if needed
    if sample_size and sample_size < len(flat_data):
        indices = np.random.choice(len(flat_data), sample_size, replace=False)
        sample_data = flat_data[indices]
    else:
        sample_data = flat_data
    
    # Chi-square attack
    chi2_stat, p_value, is_stego_likely = chi_square_attack_lsb(sample_data)
    results['chi_square'] = {
        'statistic': chi2_stat,
        'p_value': p_value,
        'is_stego_likely': is_stego_likely
    }
    
    # Sample Pair Analysis
    spa_rate = sample_pair_analysis(sample_data)
    results['sample_pair'] = {
        'embedding_rate': spa_rate
    }
    
    # RS Analysis
    rs_rate, diff, rm, sm, r_m, s_m = rs_analysis(sample_data, sample_size=min(1000, len(sample_data)//4))
    results['rs_analysis'] = {
        'embedding_rate': rs_rate,
        'difference': diff,
        'rm_ratio': rm,
        'sm_ratio': sm,
        'r_m_ratio': r_m,
        's_m_ratio': s_m
    }
    
    # Overall assessment
    detection_votes = 0
    if is_stego_likely:
        detection_votes += 1
    if spa_rate > 0.3:  # 30% threshold for SPA
        detection_votes += 1
    if rs_rate > 0.2:   # 20% threshold for RS
        detection_votes += 1
        
    results['overall'] = {
        'detection_votes': detection_votes,
        'is_stego_likely': detection_votes >= 2,  # At least 2 methods agree
        'confidence': detection_votes / 3.0      # Confidence level (0-1)
    }
    
    return results
