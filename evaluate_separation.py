# Summary:  Evaluate SDR, SIR and SAR of separated events and background. 
# Author:   Qiuqiang Kong
# Created:  2017.10
# Modified: - 
# ==============================================================================
import os
import h5py
import sys
import argparse
import yaml
import numpy as np
from mir_eval.separation import bss_eval_sources

import config as cfg
import prepare_data as pp_data

def evaluate_separation(args):
    workspace = args.workspace
    
    audio_dir = os.path.join(workspace, "mixed_audio", "testing")
    separated_dir = os.path.join(workspace, "separated_wavs", args.sep_type)
    ix_to_lb = cfg.ix_to_lb
    
    dict = {}
    for e in cfg.events + ['bg']:
        dict[e] = {'sdr_list': [], 'sir_list': [], 'sar_list': []}
    
    names = os.listdir(audio_dir)
    cnt = 0
    for na in names:
        if '.yaml' in na:
            bare_name = os.path.splitext(na)[0]
            
            # Read yaml
            yaml_path = os.path.join(audio_dir, na)
            with open(yaml_path, 'r') as f:
                data = yaml.load(f)
            event_type = data['event_type']
            
            # Read audio
            gt_audio_path = os.path.join(audio_dir, "%s.mix_0db.wav" % bare_name)
            (gt_bg_audio, gt_event_audio, _) = pp_data.read_audio_stereo(gt_audio_path)
            
            sep_bg_audio_path = os.path.join(separated_dir, "%s.sep_bg.wav" % bare_name)
            (sep_bg_audio, _) = pp_data.read_audio_sum_if_stereo(sep_bg_audio_path)
            sep_event_audio_path = os.path.join(separated_dir, "%s.sep_event.wav" % bare_name)
            (sep_event_audio, _) = pp_data.read_audio_sum_if_stereo(sep_event_audio_path)
            
            # Evaluate SDR, SIR and SAR
            gt_array = np.array((gt_bg_audio, gt_event_audio))
            sep_array = np.array((sep_bg_audio, sep_event_audio))
            
            (sdr, sir, sar, perm) = bss_eval_sources(gt_array, sep_array, compute_permutation=False)
            logging.info("%d, %s, %s" % (cnt, na, event_type))
            logging.info((sdr, sir, sar, perm))
            
            dict[event_type]['sdr_list'].append(sdr[1])
            dict[event_type]['sir_list'].append(sir[1])
            dict[event_type]['sar_list'].append(sar[1])
            dict['bg']['sdr_list'].append(sdr[0])
            dict['bg']['sir_list'].append(sir[0])
            dict['bg']['sar_list'].append(sar[0])
    
            cnt += 1
        
    avg = {}
    for e in ['sdr', 'sir', 'sar']:
        avg[e] = []
        
    for event_type in dict.keys():
        logging.info(event_type)
        for evaluate_type in dict[event_type]:
            tmp = np.mean(dict[event_type][evaluate_type])
            logging.info((evaluate_type, tmp))
            avg[evaluate_type[0:3]].append(tmp)

    logging.info("Average stats:")
    for e in ['sdr', 'sir', 'sar']:
        logging.info("%s, %f" % (e, np.mean(avg[e])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--workspace", type=str)
    parser.add_argument("--sep_type", type=str, help="The sub folder of separation. ")
    args = parser.parse_args()
    
    logs_dir = os.path.join(args.workspace, "logs", pp_data.get_filename(__file__))
    pp_data.create_folder(logs_dir)
    logging = pp_data.create_logging(logs_dir, filemode='w')
    logging.info(os.path.abspath(__file__))
    logging.info(sys.argv)
    
    evaluate_separation(args)