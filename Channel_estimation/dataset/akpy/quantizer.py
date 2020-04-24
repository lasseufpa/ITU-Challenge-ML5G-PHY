import numpy as np

def ak_quantizer(x, b, x_min, x_max, force_zero_level=0):
    # obs: in original function, x_min, x_max have default values
    M = 2 ** b #number of quantization levels
    
    # Choose the min value such that the result coincides with Lloyd's
    # optimum quantizer when the input is uniformly distributed. Instead of
    # delta=abs((xmax-xmin)/(M-1)); as quantization step use:

    delta = np.abs((x_max - x_min) / M)
    quantizer_levels = x_min + (delta / 2) + np.arange(0,M)*delta

    if force_zero_level == 1:
        zero_represented = list(quantizer_levels).count(0) #is 0 there?
        if zero_represented == 0: # zero is not represented yet
            abs_levels = np.abs(quantizer_levels)
            min_abs = np.min(abs_levels)
            # take in account that two levels, say -5 and 5 can be minimum
            # make sure it is the largest, such that there are more negative
            # quantizer levels than positive

            min_level_indices = [i for i, n in enumerate(abs_levels) if n == min_abs]
            closest_ind = min_level_indices[-1]
            closest_to_zero_value = quantizer_levels[closest_ind]
            quantizer_levels = quantizer_levels - closest_to_zero_value
        
    
    x_minq = np.min(quantizer_levels)
    x_maxq = np.max(quantizer_levels)
    x_i = (x - x_minq) / delta # quantizer levels
    x_i = np.round(x_i)
    x_i[x_i < 0] = 0
    x_i[x_i > 2 ** b - 1] = 2 ** b - 1
    
    x_q = x_i * delta + x_minq

    partitions_thresholds = 0.5 * (quantizer_levels[0:-2]) + quantizer_levels[1:-1]

    return x_q, x_i, quantizer_levels, partitions_thresholds








