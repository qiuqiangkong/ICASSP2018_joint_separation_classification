"""
Summary:  Mix audio. 
Usage:    scaler = get_amplitude_scaling_factor(event_audio, bg_audio, target_ebr_db)
          event_audio *= scaler
          (mixed_audio, event_audio, bg_audio, alpha, bgn_sample, fin_sample) = do_mixing(event_audio, bg_audio)
Author:   Qiuqiang Kong
Created:  2017.08.16
Modified: -
--------------------------------------
"""
import numpy as np

def rmse(y):
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))

def get_amplitude_scaling_factor(s, n, target_ebr_db, method='rmse'):
    """Get scaler of sourc1 from SNR. 
    
    Args:
      s: ndarray, source1
      n: ndarray, source2
      target_ebr_db: float, SNR
      method: 'rmse'
      
    Outputs:
      float, scaler
    """
    original_sn_rmse_ratio = rmse(s) / rmse(n)
    target_sn_rmse_ratio =  10 ** (target_ebr_db / float(20))
    signal_scaling_factor = target_sn_rmse_ratio/original_sn_rmse_ratio
    return signal_scaling_factor

def do_mixing(s, n):
    """Mix source1 and source2. Source1 will be centered in the mixture. 
    
    Args:
      s: ndarray, source1
      n: ndarray, source2
      
    Returns:
      mix_audio: ndarray, mixed audio
      s: ndarray, pad or truncated and scalered source1
      n: ndarray, scaled source2
      alpha: float, scaler
      bgn_sample: integar
      fin_sample: integar
    """
    if len(s) > len(n):
        bgn_sample = 0
        fin_sample = len(n)
        s = s[0 : len(n)]
        mix_audio = s + n
    else:
        n_half = int((len(n) - len(s)) / 2) + 1
        bgn_sample = n_half
        fin_sample = n_half + len(s)
        s = np.concatenate((np.zeros(n_half), s, np.zeros(n_half)))
        s = s[0 : len(n)]
        mix_audio = s + n
        
    alpha = 1. / np.max(np.abs(mix_audio))
    mix_audio *= alpha
    s *= alpha
    n *= alpha
    return mix_audio, s, n, alpha, bgn_sample, fin_sample