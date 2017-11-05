# Summary:  Joint segmentation and classification (JSC) model. 
# Author:   Qiuqiang Kong, Yong Xu
# Created:  2017.10
# Modified: - 
# ==============================================================================
import os
import sys
import time
import pickle
import cPickle
import numpy as np
import h5py
import shutil
import matplotlib.pyplot as plt
import argparse
import logging
import yaml
import theano
import theano.tensor as T

import prepare_data as pp_data
import config as cfg
from data_generator import DataGenerator

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
from hat import metrics

r = 0.999

def _seg_mask_ext_bg(input, **kwargs):
    """Add background segentation mask to segmentation masks. The segmentation 
    mask of 'background' is the residual of the sum of segmentation masks of all 
    events. 
    
    Args: 
      input: (n_clips, n_events, n_time, n_freq), segmentation masks of 
          'babycry', 'glassbreak' and 'gunshot'. 
      
    Returns: 
      output: (n_clips, n_events+1, n_time, n_freq), segmentation masks of
          'babycry', 'glassbreak', 'gunshot' and 'background'. 
    """
    _eps = 1e-7
    sum_seg_masks = T.sum(input, axis=1)    # (n_clips, n_time, n_freq)
    sum_seg_masks = T.clip(sum_seg_masks, _eps, 1. - _eps)
    bg_map = 1. - sum_seg_masks[:, None, :, :]  # (n_clips, 1, n_time, n_freq)
    output = T.concatenate((input, bg_map), axis=1) # (n_clips, n_events+1, n_time, n_freq)
    return output

def _global_rank_pooling(input, **kwargs):
    """Global rank pooling. 
    
    Args: 
      input: (n_clips, n_events+1, n_time, n_freq), segmentation masks. 
      weighted1d: 1darray, pre-computed weight values. 
      
    Returns:
      output: (n_clips, n_events+1), presence probabilites of each audio event
          and background. 
    """
    weight1d = kwargs['weight1d']   # (1, r, r^2, r^3, ...r^(n_time*n_freq))
    [n_songs, n_fmaps, n_time, n_freq] = input.shape
    input2d = input.reshape((n_songs*n_fmaps, n_time*n_freq))   # For sort on each feature map. 
    sorted2d = T.sort(input2d, axis=-1)[:, ::-1]    # Sort on each feature map in descend. 
    out2d = sorted2d * weight1d / T.sum(weight1d)   # Weighted sum. 
    out4d = out2d.reshape((n_songs, n_fmaps, n_time, n_freq))
    output = T.sum(out4d, axis=(2,3))     # Sum of each weighted feature map. 
    return output
    
def _global_max_pooling(input, **kwargs):
    """Global max pooling. 
    
    Args: 
      input: (n_clips, n_events+1, n_time, n_freq), segmentation masks. 
      
    Returns:
      output: (n_clips, n_events+1), presence probabilites of each audio event
          and background. 
    """
    return T.max(input, axis=(2,3))     # Max value of each feature map. 
    
def _global_avg_pooling(input, **kwargs):
    """Global average pooling. 
    
    Args: 
      input: (n_clips, n_events+1, n_time, n_freq), segmentation masks. 
      
    Returns:
      output: (n_clips, n_events+1), presence probabilites of each audio event
          and background. 
    """
    return T.max(input, axis=(2,3))     # Average value of each feature map. 


# Train
def train(args):
    workspace = args.workspace
    cla_mapping = args.cla_mapping
    
    # Load data. 
    t1 = time.time()
    tr_pack_path = os.path.join(workspace, "packed_features", "logmel", "training.h5")
    te_pack_path = os.path.join(workspace, "packed_features", "logmel", "testing.h5")

    with h5py.File(tr_pack_path, 'r') as hf:
        tr_na_list = list(hf.get('na_list'))
        tr_x = np.array(hf.get('x'))
        tr_y = np.array(hf.get('y'))
    
    with h5py.File(te_pack_path, 'r') as hf:
        te_na_list = list(hf.get('na_list'))
        te_x = np.array(hf.get('x'))
        te_y = np.array(hf.get('y'))
    logging.info("Loading data time: %s" % (time.time() - t1,))
    
    # Scale. 
    t1 = time.time()
    scaler_path = os.path.join(workspace, "scalers", "logmel", "training.scaler")
    scaler = pickle.load(open(scaler_path, 'rb'))
    tr_x = pp_data.do_scaler_on_x3d(tr_x, scaler)
    te_x = pp_data.do_scaler_on_x3d(te_x, scaler)
    logging.info("Scale time: %s" % (time.time() - t1,))
    
    logging.info("tr_x: %s %s" % (tr_x.shape, tr_x.dtype))
    logging.info("tr_y: %s %s" % (tr_y.shape, tr_y.dtype))
    logging.info("y: 1-of-4 representation: %s" % (cfg.events + ['bg'],))
    
    # Build model. 
    (_, n_time, n_freq) = tr_x.shape
    n_out = len(cfg.events) + 1
    
    in0 = InputLayer(in_shape=(n_time, n_freq))
    a1 = Reshape((1, n_time, n_freq))(in0)
    
    a1 = Conv2D(n_outfmaps=64, n_row=3, n_col=3, act='linear', border_mode=(1,1))(a1)
    a1 = BN(axis=(0, 2, 3))(a1)
    a1 = Activation('relu')(a1)
    a1 = Conv2D(n_outfmaps=64, n_row=3, n_col=3, act='linear', border_mode=(1,1))(a1)
    a1 = BN(axis=(0, 2, 3))(a1)
    a1 = Activation('relu')(a1)
    a1 = Dropout(0.3)(a1)
    
    a1 = Conv2D(n_outfmaps=64, n_row=3, n_col=3, act='linear', border_mode=(1,1))(a1)
    a1 = BN(axis=(0, 2, 3))(a1)
    a1 = Activation('relu')(a1)
    a1 = Conv2D(n_outfmaps=64, n_row=3, n_col=3, act='linear', border_mode=(1,1))(a1)
    a1 = BN(axis=(0, 2, 3))(a1)
    a1 = Activation('relu')(a1)
    a1 = Dropout(0.3)(a1)
    
    a1 = Conv2D(n_outfmaps=64, n_row=3, n_col=3, act='linear', border_mode=(1,1))(a1)
    a1 = BN(axis=(0, 2, 3))(a1)
    a1 = Activation('relu')(a1)
    a1 = Conv2D(n_outfmaps=64, n_row=3, n_col=3, act='linear', border_mode=(1,1))(a1)
    a1 = BN(axis=(0, 2, 3))(a1)
    a1 = Activation('relu')(a1)
    a1 = Dropout(0.3)(a1)
    
    a1 = Conv2D(n_outfmaps=64, n_row=3, n_col=3, act='linear', border_mode=(1,1))(a1)
    a1 = BN(axis=(0, 2, 3))(a1)
    a1 = Activation('relu')(a1)
    a1 = Conv2D(n_outfmaps=64, n_row=3, n_col=3, act='linear', border_mode=(1,1))(a1)
    a1 = BN(axis=(0, 2, 3))(a1)
    a1 = Activation('relu')(a1)
    a1 = Dropout(0.3)(a1)

    # Segmentation mask for 'babycry', 'glassbreak' and 'gunshot'. 
    a1 = Conv2D(n_outfmaps=len(cfg.events), n_row=1, n_col=1, act='sigmoid', border_mode=(0,0))(a1)
    
    # Extend segmentation mask to 'babycry', 'glassbreak', 'gunshot' and 'background'. 
    a1 = Lambda(_seg_mask_ext_bg, name='seg_masks')(a1)
    
    # Classification mapping. 
    cla_mapping = args.cla_mapping
    
    if cla_mapping == 'global_rank_pooling':
        weight1d = np.power(r * np.ones(120*64), np.arange(120*64))
        a8 = Lambda(_global_rank_pooling, weight1d=weight1d, name='a5')(a1)
    elif cla_mapping == 'global_max_pooling':
        a8 = Lambda(_global_max_pooling)(a1)
    elif cla_mapping == 'global_avg_pooling':
        a8 = Lambda(_global_avg_pooling)(a1)
    else:
        raise Exception("Incorrect cla_mapping!")
    
    md = Model([in0], [a8])
    md.compile()
    md.summary(is_logging=True)
    
    # Callbacks. 
    md_dir = os.path.join(workspace, "models", pp_data.get_filename(__file__))
    pp_data.create_folder(md_dir)
    save_model = SaveModel(md_dir, call_freq=100, type='iter')
    validation = Validation(te_x=te_x, te_y=te_y, 
                            batch_size=100, 
                            call_freq=50, 
                            metrics=['binary_crossentropy'], 
                            dump_path=None, 
                            is_logging=True)
    callbacks = [save_model, validation]
    
    # Train. 
    generator = DataGenerator(batch_size=20, type='train')
    loss_ary = []
    t1 = time.time()
    optimizer = Adam(1e-4)
    for (batch_x, batch_y) in generator.generate(xs=[tr_x], ys=[tr_y]):
        np.set_printoptions(threshold=np.nan, linewidth=1000, precision=2, suppress=True)
        loss = md.train_on_batch(batch_x, batch_y, 
                                 loss_func='binary_crossentropy', 
                                 optimizer=optimizer, 
                                 callbacks=callbacks)
        loss_ary.append(loss)
        if md.iter_ % 50 == 0:  # Evalute training loss every several iterations. 
            logging.info("iter: %d, tr loss: %d" % (md.iter_, np.mean(loss_ary)))
            logging.info("time: %s" % (time.time() - t1,))
            t1 = time.time()
            loss_ary = []
        if md.iter_ == 10001:   # Stop after several iterations. 
            break
    
    
# Recognize on testing data. 
def recognize(args):
    workspace = args.workspace
    md_path = os.path.join(workspace, "models", pp_data.get_filename(__file__), 
                           args.model_name)
    t1 = time.time()
    
    # Load scaler. 
    scaler_path = os.path.join(workspace, "scalers", "logmel", "training.scaler")
    scaler = pickle.load(open(scaler_path, 'rb'))
    
    # Load model. 
    md = serializations.load(md_path)
    
    # Observe function. 
    observe_nodes = [md.find_layer('seg_masks').output_]
    f_forward = md.get_observe_forward_func(observe_nodes)
    
    audio_dir = os.path.join(workspace, "mixed_audio", "testing")
    names = os.listdir(audio_dir)
    
    at_pd_ary = []
    at_gt_ary = []
    sed_pd_ary = []
    sed_gt_ary = []
    
    # For all audio clips. 
    for na in names:
        if '.mix_0db.wav' in na:
            logging.info(na)
            
            # Load audio. 
            bare_na = os.path.splitext(os.path.splitext(na)[0])[0]
            audio_path = os.path.join(audio_dir, na)
            (bg_audio, event_audio, fs) = pp_data.read_audio_stereo(audio_path)
            mixed_audio = bg_audio + event_audio
            
            # Load yaml. 
            yaml_path = os.path.join(audio_dir, "%s.yaml" % bare_na)
            with open(yaml_path, 'r') as f:
                data = yaml.load(f)
            event_type = data['event_type']
            
            # Calculate feature. 
            x = pp_data.calc_feat(mixed_audio)
            x3d = pp_data.do_scaler_on_x3d(x[np.newaxis, ...], scaler)

            # Ground truth. 
            gt_y = [0, 0, 0, 0]
            gt_y[cfg.lb_to_ix[event_type]] = 1
            at_gt_ary.append(gt_y)

            # Audio tagging (AT) prediction. 
            [pred_y] = md.predict(x3d)  # (1, n_events+1)
            pred_y = pred_y[0]      # (n_events+1,)
            at_pd_ary.append(pred_y)
            
            # Sound event detection (SED) prediction. 
            [masks] = md.run_function(f_forward, x3d, batch_size=10, tr_phase=0.)   # (1, n_events+1, n_time, n_freq)
            masks = masks[0]    # (n_events+1, n_time, n_freq)
            sed_pd = np.mean(masks, axis=-1).T  # (n_time, n_events+1)
            sed_pd_ary.append(sed_pd)
            sed_gt = np.zeros_like(sed_pd)
            [bgn_sec, fin_sec] = data['event_segment']
            bgn_fr = int(bgn_sec * cfg.sample_rate / float(cfg.n_window - cfg.n_overlap))
            fin_fr = int(fin_sec * cfg.sample_rate / float(cfg.n_window - cfg.n_overlap))
            sed_gt[bgn_fr : fin_fr, cfg.lb_to_ix[event_type]] = 1
            sed_gt_ary.append(sed_gt)
    
    at_pd_ary = np.array(at_pd_ary)
    at_gt_ary = np.array(at_gt_ary)
    sed_pd_ary = np.array(sed_pd_ary)
    sed_gt_ary = np.array(sed_gt_ary)
    
    # Write out AT and SED presence probabilites. 
    logging.info("at_pd_ary.shape: %s" % (at_pd_ary.shape,))
    logging.info("at_gt_ary.shape: %s" % (at_gt_ary.shape,))
    logging.info("sed_pd_ary.shape: %s" % (sed_pd_ary.shape,))
    logging.info("sed_gt_ary.shape: %s" % (sed_gt_ary.shape,))
    dict = {}
    dict['at_pd_ary'] = at_pd_ary
    dict['at_gt_ary'] = at_gt_ary
    dict['sed_pd_ary'] = sed_pd_ary
    dict['sed_gt_ary'] = sed_gt_ary
    out_path = os.path.join(workspace, "_tmp", "_at_sed_dict.p")
    pp_data.create_folder(os.path.dirname(out_path))
    cPickle.dump(dict, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    logging.info("Recognize time: %s" % (time.time() - t1,))


# Get stats. 
def get_stats(args):
    workspace = args.workspace
    
    # Load AT and SED presence probabilites. 
    pickle_path = os.path.join(workspace, "_tmp", "_at_sed_dict.p")
    dict = cPickle.load(open(pickle_path, 'rb'))
    at_pd_ary = dict['at_pd_ary']
    at_gt_ary = dict['at_gt_ary']
    sed_pd_ary = dict['sed_pd_ary']
    sed_gt_ary = dict['sed_gt_ary']
    
    # AT eer. 
    logging.info("--- AT EER ---")
    for i1 in xrange(len(cfg.events)):
        eer = pp_data.compute_eer(at_pd_ary[:, i1], at_gt_ary[:, i1])
        logging.info("%s: %f" % (cfg.events[i1], eer))
        
    # SED eer. 
    logging.info("--- SED EER ---")
    for i1 in xrange(len(cfg.events)):
        eer = pp_data.compute_eer(sed_pd_ary[:, :, i1].flatten(), sed_gt_ary[:, :, i1].flatten())
        logging.info("%s: %f" % (cfg.events[i1], eer))
    
    
# Plot segmentation masks. 
def plot_seg_masks(args):
    # Load data. 
    te_pack_path = os.path.join(workspace, "packed_features", "logmel", "testing.h5")
    scaler_path = os.path.join(workspace, "scalers", "logmel", "training.scaler")
    
    with h5py.File(te_pack_path, 'r') as hf:
        te_na_list = list(hf.get('na_list'))
        te_x = np.array(hf.get('x'))
        te_y = np.array(hf.get('y'))
        
    te_x_unscaled = te_x  # unscaled x for plot. 
    scaler = pickle.load(open(scaler_path, 'rb'))
    te_x = pp_data.do_scaler_on_x3d(te_x, scaler)
    
    # Load model. 
    md_path = os.path.join(workspace, "models", pp_data.get_filename(__file__), 
                           args.model_name)
    md = serializations.load(md_path)
    
    # Observe function. 
    observe_nodes = [md.find_layer('seg_masks').output_]
    f_forward = md.get_observe_forward_func(observe_nodes)
    [seg_masks] = md.run_function(f_forward, te_x, batch_size=50, tr_phase=0.)
    print("Segmentation masks: %s" % (seg_masks.shape,))
    
    # Plot segmentation masks. 
    for i1 in xrange(len(seg_masks)):
        na = te_na_list[i1]
        if ".mix_0db.wav" in na:
            print(na)
            gt_y = te_y[i1].astype(np.float32)
            print(gt_y)
            print("Ground truth: %s" % cfg.events[np.argmax(gt_y)])
            
            events_ex = cfg.events + ['bg']
            fig, axs = plt.subplots(3,2, sharex=True)
            axs[0, 0].matshow(te_x_unscaled[i1].T, origin='lower', aspect='auto')
            axs[0, 0].set_title("log Mel spectrogram")
            for i2 in xrange(0, 4):
                axs[i2/2+1, i2%2].matshow(seg_masks[i1, i2].T, origin='lower', aspect='auto', vmin=0, vmax=1)
                axs[i2/2+1, i2%2].set_title(events_ex[i2])
            plt.show()
    
    
if __name__ == '__main__':
    # Parse arguments. 
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str)
    parser_train.add_argument('--cla_mapping', type=str, 
        choices=['global_rank_pooling', 'global_max_pooling', 'global_avg_pooling'])
    
    parser_recognize = subparsers.add_parser('recognize')
    parser_recognize.add_argument('--workspace', type=str)
    parser_recognize.add_argument('--model_name', type=str, help="E.g. md10000_iters.p")
    
    parser_get_stats = subparsers.add_parser('get_stats')
    parser_get_stats.add_argument('--workspace', type=str)

    parser_hotmap = subparsers.add_parser('plot_seg_masks')
    parser_hotmap.add_argument('--workspace', type=str)
    parser_hotmap.add_argument('--model_name', type=str)
    
    args = parser.parse_args()
    
    # Logs. 
    workspace = args.workspace
    logs_dir = os.path.join(workspace, "logs", pp_data.get_filename(__file__))
    pp_data.create_folder(logs_dir)
    logging = pp_data.create_logging(logs_dir, filemode='w')
    logging.info(os.path.abspath(__file__))
    logging.info(sys.argv)
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'recognize':
        recognize(args)
    elif args.mode == 'get_stats':
        get_stats(args)
    elif args.mode == 'plot_seg_masks':
        plot_seg_masks(args)
    else:
        raise Exception("Wrong!")