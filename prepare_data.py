from __future__ import print_function
import numpy as np
import os
import yaml
import sys
import soundfile
import pickle
import cPickle
import logging
import time
import librosa
from sklearn import metrics
import h5py
from scipy import signal
from sklearn import preprocessing
import argparse

from mix_audio import mix_audio
import config as cfg


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def get_filename(path):
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na

def read_meta_yaml(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f)
    return data
    
def read_audio_stereo(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    audio_1 = audio[:, 0]
    audio_2 = audio[:, 1]
    if target_fs is not None and fs != target_fs:
        audio_1 = librosa.resample(audio_1, orig_sr=fs, target_sr=target_fs)
        audio_2 = librosa.resample(audio_2, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio_1, audio_2, fs

def read_audio_sum_if_stereo(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.sum(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs
    
def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs
    
def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

def write_meta_yaml(filename, data):
    with open(filename, 'w') as f:
        f.write(yaml.dump(data, default_flow_style=False))

def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na

# Write out to file
def create_logging(log_dir, filemode):
    # Write out to file
    i1 = 0
    while os.path.isfile(os.path.join(log_dir, "%05d.log" % i1)):
        i1 += 1
    log_path = os.path.join(log_dir, "%05d.log" % i1)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)
                
    # Print to console   
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging

###
def batch_mix(dataset_dir, mixed_yaml_path, target_ebr_db_ary, out_dir):
    """Create mixed audio waveform from yaml file. 
    
    Args:
      dataset_dir: string, dataset directory. 
      mixed_yaml_path: string, path of yaml file containing mixed information. 
      target_ebr_db_ary: list of float, audio with these SDRs will be created. 
      out_dir: string, path of writing out directory. 
      
    Returns: 
      None. 
    """
    fs = cfg.sample_rate
    mixed_yaml_data = read_meta_yaml(mixed_yaml_path)
    create_folder(out_dir)

    t1 = time.time()
    cnt = 0
    for i1 in xrange(len(mixed_yaml_data)):
        # Read yaml
        event_name = mixed_yaml_data[i1]['event_name']
        event_type = mixed_yaml_data[i1]['event_type']
        [event_bgn_sec_origin, event_fin_sec_origin] = mixed_yaml_data[i1]['event_segment']
        bg_name = mixed_yaml_data[i1]['bg_name']
        bg_type = mixed_yaml_data[i1]['bg_type']
        [bg_bgn_sec, bg_fin_sec] = mixed_yaml_data[i1]['bg_segment']
        print(cnt, event_name, event_bgn_sec_origin, event_fin_sec_origin)
        
        event_path = os.path.join(dataset_dir, 'events', event_type, event_name)
        bg_path = os.path.join(dataset_dir, 'bgs', bg_name)
        
        # Read audio
        (event_audio, _) = read_audio(event_path, fs)
        (bg_audio, _) = read_audio(bg_path, fs)
        event_audio = event_audio[int(event_bgn_sec_origin * fs) : int(event_fin_sec_origin * fs)]
        bg_audio = bg_audio[int(bg_bgn_sec * fs) : int(bg_fin_sec * fs)]
        
        # Write out clean bg
        out_bg_audio_path = os.path.join(out_dir, "%04d.bg.wav" % cnt)
        bg_audio_scalered = bg_audio / np.max(np.abs(bg_audio))
        write_audio(out_bg_audio_path, bg_audio_scalered, fs)
        
        # Write out clean centered event
        out_event_audio_path = os.path.join(out_dir, "%04d.event.wav" % cnt)
        sil_audio = np.zeros_like(bg_audio)
        (_, event_audio_scalered, _, _, event_bgn_sample, event_fin_sample) = mix_audio.do_mixing(event_audio, sil_audio)   # center the event
        event_audio_scalered = event_audio_scalered / np.max(np.abs(event_audio_scalered))
        write_audio(out_event_audio_path, event_audio_scalered, fs)
        
        # Write out mixed audio
        for db in target_ebr_db_ary:
            out_mixed_audio_path = os.path.join(out_dir, "%04d.mix_%ddb.wav" % (cnt, db))
            scaler = mix_audio.get_amplitude_scaling_factor(event_audio, bg_audio, db)
            event_audio_scalered = scaler * event_audio     # scaler event
            (_, event_audio_scalered, bg_audio_scalered, alpha, _, _) = mix_audio.do_mixing(event_audio_scalered, bg_audio)
            mixed_audio = np.array((bg_audio_scalered, event_audio_scalered)).T
            write_audio(out_mixed_audio_path, mixed_audio, fs)
            
        # Write yaml
        event_bgn_sec = event_bgn_sample / float(fs)
        event_fin_sec = event_fin_sample / float(fs)
        
        out_yaml_path = os.path.join(out_dir, "%04d.yaml" % cnt)
        out_yaml = {'event_name': event_name, 
                    'event_type': event_type, 
                    'event_origin_segment': [event_bgn_sec_origin, event_fin_sec_origin], 
                    'event_segment': [event_bgn_sec, event_fin_sec], 
                    'bg_audio': bg_name, 
                    'bg_type': bg_type, 
                    'bg_segment': [bg_bgn_sec, bg_fin_sec], 
                    }
        write_meta_yaml(out_yaml_path, out_yaml)
        cnt += 1
        
    print("Batch mix finished!", time.time() - t1, "s")
        
###
def calculate_logmel(audio_dir, out_dir):
    """Calculate log Mel feature for each audio and write out to cPickle. 
    
    Args: 
      audio_dir: string, directory of audio clips. 
      out_dir: string, directory to write out features. 
      
    Returns: 
      None. 
    """
    create_folder(out_dir)

    t1 = time.time()
    names = os.listdir(audio_dir)
    names = sorted(names)
    for na in names:
        if na.endswith('.wav'):
            audio_path = os.path.join(audio_dir, na)
            (audio, _) = read_audio_sum_if_stereo(audio_path)
            x = calc_feat(audio)            
            out_path = os.path.join(out_dir, os.path.splitext(na)[0] + ".p")
            cPickle.dump(x, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    print("Calculate logmel finished!", time.time() - t1, "s")
    
def calc_feat(audio):
    """Calculate log Mel feature for an audio array. 
    
    Args: 
      audio: 1d-array. 
      
    Returns: 
      x: 2d-array, log Mel spectrogram, (n_time, n_freq). 
    """
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    ham_win = np.hamming(n_window)
    
    [f, t, x] = signal.spectral.spectrogram(
                    audio, 
                    window=ham_win,
                    nperseg=n_window, 
                    noverlap=n_overlap, 
                    detrend=False, 
                    return_onesided=True, 
                    mode='magnitude') 
    x = x.T
    if globals().get('melW') is None:
        global melW
        melW = librosa.filters.mel(sr=fs, 
                                n_fft=n_window, 
                                n_mels=64, 
                                fmin=0., 
                                fmax=fs / 2)
    x = np.dot(x, melW.T)
    x = np.log(x + 1e-8)
    x = x.astype(np.float32)
    return x

###    
def pack_features(fe_dir, yaml_dir, out_path):
    """Pack feature files to a single hdf5 file. 
    
    Args: 
      fe_dir: string, path of feature directory. 
      yaml_dir: string, directory of yaml file of mixed audio. 
      out_path: string, path of hdf5 to write out. 
      
    Returns: 
      None. 
    """
    create_folder(os.path.dirname(out_path))
    names = os.listdir(fe_dir)
    names = sorted(names)
    lb_to_ix = cfg.lb_to_ix
    
    t1 = time.time()
    x_all, y_all, na_list = [], [], []
    
    names = os.listdir(yaml_dir)
    for na in names:
        if os.path.splitext(na)[1] == ".yaml":
            bare_name = os.path.splitext(na)[0]
            
            # Read yaml
            yaml_path = os.path.join(yaml_dir, na)
            with open(yaml_path, 'r') as f:
                data = yaml.load(f)
            event_type = data['event_type']
            
            # Background
            bg_fe_path = os.path.join(fe_dir, "%s.bg.p" % bare_name)
            bg_x = cPickle.load(open(bg_fe_path, 'rb'))
            x_all.append(bg_x)
            y = [0, 0, 0, 1]
            y_all.append(y)
            na_list.append("%s.bg.wav" % bare_name)
            
            # Event
            event_fe_path = os.path.join(fe_dir, "%s.event.p" % bare_name)
            event_x = cPickle.load(open(event_fe_path, 'rb'))
            x_all.append(event_x)
            ix = cfg.lb_to_ix[event_type]
            y = [0, 0, 0, 0]
            y[ix] = 1
            y_all.append(y)
            na_list.append("%s.event.wav" % bare_name)
            
            # Mixture
            mixture_fe_path = os.path.join(fe_dir, "%s.mix_0db.p" % bare_name)
            mixture_x = cPickle.load(open(mixture_fe_path, 'rb'))
            x_all.append(mixture_x)
            ix = cfg.lb_to_ix[event_type]
            y = [0, 0, 0, 1]
            y[ix] = 1
            y_all.append(y)
            na_list.append("%s.mix_0db.wav" % bare_name)
            
    x_all = np.array(x_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)
    
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('x', data=x_all)
        hf.create_dataset('y', data=y_all)
        hf.create_dataset('na_list', data=na_list)
    
    print("Pack features time: %s" % (time.time() - t1,))
        
###
def compute_scaler(pack_path, out_path): 
    """Compute scaler of each feature channel and write out. 
    
    Args: 
      pack_path: string, path of packed hdf5 file. 
      out_path: string, path to write out the scaler. 
      
    Returns: 
      None. 
    """
    create_folder(os.path.dirname(out_path))
    with h5py.File(pack_path, 'r') as hf:
        x = np.array(hf.get('x'))
    
    print(x.shape)
    (N, n_time, n_freq) = x.shape
    x = x.reshape((N * n_time, n_freq))
    
    t1 = time.time()
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(x)
    print("time:", time.time() - t1)
    print("Mean: ", scaler.mean_)
    print("Scale: ", scaler.scale_)
    pickle.dump(scaler, open(out_path, 'wb'))
    print("Compute scaler finished!")

def do_scaler_on_x3d(x, scaler):
    """Use scaler to scale input. 
    
    Args:
      x: (n_clips, n_time, n_freq). 
      scaler: object. 
      
    Returns:
      x3d: (n_clips, n_time, n_freq), scaled input. 
    """
    (N, n_time, n_freq) = x.shape
    x2d = x.reshape((N * n_time, n_freq))
    x2d = scaler.transform(x2d)
    x3d = x2d.reshape((N, n_time, n_freq))
    return x3d

###
def compute_eer(pred, gt):
    fpr, tpr, thresholds = metrics.roc_curve(gt, pred, drop_intermediate=True)
    
    eps = 1E-6
    Points = [(0,0)]+zip(fpr, tpr)
    for i, point in enumerate(Points):
        if point[0]+eps >= 1-point[1]:
            break
    P1 = Points[i-1]; P2 = Points[i]
        
    #Interpolate between P1 and P2
    if abs(P2[0]-P1[0]) < eps:
        EER = P1[0]        
    else:        
        m = (P2[1]-P1[1]) / (P2[0]-P1[0])
        o = P1[1] - m * P1[0]
        EER = (1-o) / (1+m)  
    return EER

###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_mix = subparsers.add_parser('mix')
    parser_mix.add_argument('--dataset_dir', type=str)
    parser_mix.add_argument('--workspace', type=str)
    
    parser_calculate_logmel = subparsers.add_parser('calculate_logmel')
    parser_calculate_logmel.add_argument('--workspace', type=str)
    
    parser_pack_features = subparsers.add_parser('pack_features')
    parser_pack_features.add_argument('--workspace', type=str)
    
    parser_compute_scaler = subparsers.add_parser('compute_scaler')
    parser_compute_scaler.add_argument('--workspace', type=str)
    
    args = parser.parse_args()
    
    # Create mixed audio clips. 
    if args.mode == 'mix': 
        dataset_dir = args.dataset_dir
        workspace = args.workspace
        batch_mix(dataset_dir=dataset_dir, 
                  mixed_yaml_path=os.path.join(workspace, "mixed_yaml", "training.csv"), 
                  target_ebr_db_ary=[0], 
                  out_dir=os.path.join(workspace, "mixed_audio", "training"))
        batch_mix(dataset_dir=dataset_dir, 
                  mixed_yaml_path=os.path.join(workspace, "mixed_yaml", "testing.csv"), 
                  target_ebr_db_ary=[0], 
                  out_dir=os.path.join(workspace, "mixed_audio", "testing"))
    # Calculate log Mel feature for all audio clips. 
    elif args.mode == 'calculate_logmel':
        workspace = args.workspace
        calculate_logmel(audio_dir=os.path.join(workspace, "mixed_audio", "training"), 
                         out_dir=os.path.join(workspace, "features", "logmel", "training"))
        calculate_logmel(audio_dir=os.path.join(workspace, "mixed_audio", "testing"), 
                         out_dir=os.path.join(workspace, "features", "logmel", "testing"))
    # Pack feature files to a hdf5 file. 
    elif args.mode == 'pack_features':
        workspace = args.workspace
        pack_features(fe_dir=os.path.join(workspace, "features", "logmel", "training"), 
                      yaml_dir=os.path.join(workspace, "mixed_audio", "training"), 
                      out_path=os.path.join(workspace, "packed_features", "logmel", "training.h5"))
        pack_features(fe_dir=os.path.join(workspace, "features", "logmel", "testing"), 
                      yaml_dir=os.path.join(workspace, "mixed_audio", "testing"), 
                      out_path=os.path.join(workspace, "packed_features", "logmel", "testing.h5"))
    # Compute scaler of features on training data. 
    elif args.mode == 'compute_scaler':
        workspace = args.workspace
        compute_scaler(pack_path=os.path.join(workspace, "packed_features", "logmel", "training.h5"), 
                       out_path=os.path.join(workspace, "scalers", "logmel", "training.scaler"))
    else: 
        raise Exception("Incorrect argv!")