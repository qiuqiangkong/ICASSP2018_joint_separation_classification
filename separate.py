# Summary:  Do source separation using trained model. 
# Author:   Qiuqiang Kong
# Created:  2017.10
# Modified: - 
# ==============================================================================
import os
import sys
import time
import pickle
import cPickle
import numpy as np
import yaml
import h5py
import shutil
import matplotlib.pyplot as plt
import argparse
from scipy import signal
import theano
import theano.tensor as T
import librosa

import prepare_data as pp_data
import config as cfg
from spectrogram_to_wave import spectrogram_to_wave
from main import (_global_max_pooling, _global_avg_pooling, _global_rank_pooling, 
                  _seg_mask_ext_bg)

from hat.layers.core import *
from hat.layers.cnn import *
from hat.layers.pooling import *
from hat.optimizers import Adam
from hat.callbacks import SaveModel, Validation
from hat.models import Model, Sequential
from hat.preprocessing import sparse_to_categorical
from hat.layers.normalization import *
import hat.objectives as obj
from hat import activations
from hat import serializations


def get_inverse_W(W):
    return W / (np.sum(W, axis=0) + 1e-8)

def no_separation(args):
    """Write out un-separated mixture as baseline. 
    """
    workspace = args.workspace
    
    out_dir = os.path.join(workspace, "separated_wavs", "no_separation")
    pp_data.create_folder(out_dir)
    
    audio_dir = os.path.join(workspace, "mixed_audio", "testing")
    names = os.listdir(audio_dir)
    
    for na in names:
        if '.mix_0db.wav' in na:
            print(na)
            audio_path = os.path.join(audio_dir, na)
            (bg_audio, event_audio, fs) = pp_data.read_audio_stereo(audio_path)
            mixed_audio = bg_audio + event_audio
            
            bare_na = os.path.splitext(os.path.splitext(na)[0])[0]
            pp_data.write_audio(os.path.join(out_dir, bare_na + ".sep_bg.wav"), mixed_audio, fs)
            pp_data.write_audio(os.path.join(out_dir, bare_na + ".sep_event.wav"), mixed_audio, fs)
            
    print("Write out finished!")

def ibm_separation(args):
    """Ideal binary mask (IBM) source separation. 
    """
    workspace = args.workspace
    
    out_dir = os.path.join(workspace, "separated_wavs", "ibm_separation")
    pp_data.create_folder(out_dir)
    
    audio_dir = os.path.join(workspace, "mixed_audio", "testing")
    names = os.listdir(audio_dir)
    
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    clip_sec = cfg.clip_sec
    
    ham_win = np.hamming(n_window)
    recover_scaler = np.sqrt((ham_win**2).sum())
    
    for na in names:
        if '.mix_0db.wav' in na:
            print(na)
            bare_na = os.path.splitext(os.path.splitext(na)[0])[0]
            audio_path = os.path.join(audio_dir, na)
            (bg_audio, event_audio, fs) = pp_data.read_audio_stereo(audio_path)
            mixed_audio = bg_audio + event_audio

            [f, t, bg_spec] = signal.spectral.spectrogram(x=bg_audio, 
                                                    window=ham_win, 
                                                    nperseg=n_window, 
                                                    noverlap=n_overlap, 
                                                    detrend=False, 
                                                    return_onesided=True, 
                                                    scaling='density', 
                                                    mode='magnitude') 
                                                    
            [f, t, event_spec] = signal.spectral.spectrogram(x=event_audio, 
                                                    window=ham_win, 
                                                    nperseg=n_window, 
                                                    noverlap=n_overlap, 
                                                    detrend=False, 
                                                    return_onesided=True, 
                                                    scaling='density', 
                                                    mode='magnitude') 
            
            [f, t, mixed_spec] = signal.spectral.spectrogram(x=mixed_audio, 
                                                    window=ham_win, 
                                                    nperseg=n_window, 
                                                    noverlap=n_overlap, 
                                                    detrend=False, 
                                                    return_onesided=True, 
                                                    scaling='density', 
                                                    mode='complex') 
               
            bg_spec = bg_spec.T
            event_spec = event_spec.T
            mixed_spec = mixed_spec.T
                             
            ratio = 1.7     # 5 dB
            event_mask = (np.sign(event_spec / (bg_spec * ratio) -1) + 1) / 2
            bg_mask = 1. - event_mask
            
            bg_separated_spec = np.abs(mixed_spec) * bg_mask
            event_separated_spec = np.abs(mixed_spec) * event_mask

            # Write out separated music
            s = spectrogram_to_wave.recover_wav(bg_separated_spec, mixed_spec, n_overlap=n_overlap, winfunc=np.hamming, wav_len=int(fs * clip_sec))
            s *= recover_scaler
            pp_data.write_audio(os.path.join(out_dir, bare_na + ".sep_bg.wav"), s, fs)
                
            # Write out separated vocal
            s = spectrogram_to_wave.recover_wav(event_separated_spec, mixed_spec, n_overlap=n_overlap, winfunc=np.hamming, wav_len=int(fs * clip_sec))
            s *= recover_scaler
            pp_data.write_audio(os.path.join(out_dir, bare_na + ".sep_event.wav"), s, fs)
            
    print("Finished!")

def jsc_separation(args):
    """Joing separation-classification (JSC) source separation. 
    """
    workspace = args.workspace
    
    scaler_path = os.path.join(workspace, "scalers", "logmel", "training.scaler")
    scaler = pickle.load(open(scaler_path, 'rb'))
    
    md_path = os.path.join(workspace, "models", "main", args.model_name)
    md = serializations.load(md_path)
    
    out_dir = os.path.join(workspace, "separated_wavs", "jsc_separation")
    pp_data.create_folder(out_dir)
    
    observe_nodes = [md.find_layer('seg_masks').output_]
    f_forward = md.get_observe_forward_func(observe_nodes)

    audio_dir = os.path.join(os.path.join(workspace, "mixed_audio", "testing"))
    names = os.listdir(audio_dir)
    
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    ham_win = np.hamming(n_window)
    recover_scaler = np.sqrt((ham_win**2).sum())
    
    melW = librosa.filters.mel(sr=fs, 
                                n_fft=n_window, 
                                n_mels=64, 
                                fmin=0., 
                                fmax=fs / 2)
    inverse_melW = get_inverse_W(melW)
    
    for na in names:
        if ".mix" in na:
            # Read yaml
            bare_name = os.path.splitext(os.path.splitext(na)[0])[0]
            yaml_path = os.path.join(audio_dir, "%s.yaml" % bare_name)
            with open(yaml_path, 'r') as f:
                data = yaml.load(f)
            event_type = data['event_type']
            print(na, event_type)

            # Read audio
            audio_path = os.path.join(audio_dir, na)
            (bg_audio, event_audio, _) = pp_data.read_audio_stereo(audio_path)
            mixed_audio = bg_audio + event_audio
            
            # Spectrogram
            [f, t, bg_spec] = signal.spectral.spectrogram(x=bg_audio, 
                                                    window=ham_win, 
                                                    nperseg=n_window, 
                                                    noverlap=n_overlap, 
                                                    detrend=False, 
                                                    return_onesided=True, 
                                                    scaling='density', 
                                                    mode='complex') 
                                                    
            [f, t, event_spec] = signal.spectral.spectrogram(x=event_audio, 
                                                    window=ham_win, 
                                                    nperseg=n_window, 
                                                    noverlap=n_overlap, 
                                                    detrend=False, 
                                                    return_onesided=True, 
                                                    scaling='density', 
                                                    mode='complex') 
                   
            [f, t, mixed_spec] = signal.spectral.spectrogram(x=mixed_audio, 
                                                    window=ham_win, 
                                                    nperseg=n_window, 
                                                    noverlap=n_overlap, 
                                                    detrend=False, 
                                                    return_onesided=True, 
                                                    scaling='density', 
                                                    mode='complex') 
                                                    
            bg_spec = bg_spec.T
            event_spec = event_spec.T
            mixed_spec = mixed_spec.T
    
            # Log Mel spectrogram
            mixed_x = pp_data.calc_feat(mixed_audio)
            x3d = pp_data.do_scaler_on_x3d(mixed_x[np.newaxis, ...], scaler)
            
            # Segmentation masks
            [mel_masks] = md.run_function(f_forward, x3d, batch_size=10, tr_phase=0.)
            mel_masks = mel_masks[0]    # (n_time, 64)
            spec_masks = np.dot(mel_masks, inverse_melW)  # (n_time, 513)
            
            if args.plot_only:
                mixed_mel_spec = np.dot(np.abs(mixed_spec), melW.T)
                bg_mel_spec = np.dot(np.abs(bg_spec), melW.T)
                event_mel_spec = np.dot(np.abs(event_spec), melW.T)
                ratio = 1.7     # 5 dB
                event_mask = (np.sign(event_mel_spec / (bg_mel_spec * ratio) -1) + 1) / 2
                
                fig, axs = plt.subplots(3,2, sharex=True)
                axs[0, 0].matshow(np.log(mixed_mel_spec.T), origin='lower', aspect='auto')
                axs[0, 1].matshow(event_mask.T, origin='lower', aspect='auto')
                axs[1, 0].matshow(spec_masks[0].T, origin='lower', aspect='auto', vmin=0., vmax=1.)
                axs[1, 1].matshow(spec_masks[1].T, origin='lower', aspect='auto', vmin=0., vmax=1.)
                axs[2, 0].matshow(spec_masks[2].T, origin='lower', aspect='auto', vmin=0., vmax=1.)
                axs[2, 1].matshow(spec_masks[3].T, origin='lower', aspect='auto', vmin=0., vmax=1.)
                axs[0, 0].set_title('log Mel of mixture')
                axs[0, 1].set_title('IBM of event')
                axs[1, 0].set_title('babycry')
                axs[1, 1].set_title('glassbreak')
                axs[2, 0].set_title('gunshot')
                axs[2, 1].set_title('bg')
                
                plt.show()
            
            else:
                # Separated spec
                separated_specs = spec_masks * np.abs(mixed_spec)[None, :, :]
    
                # Write out all events and bg
                enlarged_events = cfg.events + ['bg']
                for i1 in xrange(4):
                    s = spectrogram_to_wave.recover_wav(separated_specs[i1], mixed_spec, n_overlap=n_overlap, winfunc=np.hamming, wav_len=len(mixed_audio))
                    s *= recover_scaler
                    pp_data.write_audio(os.path.join(out_dir, "%s.sep_%s.wav" % (bare_name, enlarged_events[i1])), s, fs)
                    
                # Write out event
                s = spectrogram_to_wave.recover_wav(separated_specs[cfg.lb_to_ix[event_type]], mixed_spec, n_overlap=n_overlap, winfunc=np.hamming, wav_len=len(mixed_audio))
                s *= recover_scaler
                pp_data.write_audio(os.path.join(out_dir, "%s.sep_event.wav" % bare_name), s, fs)
                    
                # Write out origin mix
                pp_data.write_audio(os.path.join(out_dir, "%s.sep_mix.wav" % bare_name), mixed_audio, fs)
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_no_sep = subparsers.add_parser('no_separation')
    parser_no_sep.add_argument('--workspace', type=str)
    
    parser_ibm_sep = subparsers.add_parser('ibm_separation')
    parser_ibm_sep.add_argument('--workspace', type=str)
    
    parse_do_sep = subparsers.add_parser('jsc_separation')
    parse_do_sep.add_argument('--workspace', type=str)
    parse_do_sep.add_argument('--model_name', type=str)
    parse_do_sep.add_argument('--plot_only', type=bool, default=False)
    
    args = parser.parse_args()
    
    if args.mode == 'no_separation':
        no_separation(args)
    elif args.mode == 'ibm_separation':
        ibm_separation(args)
    elif args.mode == 'jsc_separation':
        jsc_separation(args)
    else:
        raise Exception("Incorrect argument! ")