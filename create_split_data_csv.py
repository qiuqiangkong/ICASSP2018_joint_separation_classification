# Summary:  Create yaml file containing mixed information for training and testing. 
# Author:   Qiuqiang Kong
# Created:  2017.10
# Modified: - 
# ==============================================================================
import os
import yaml
import numpy as np
import csv
import argparse

import config as cfg

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def read_meta_yaml(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f)
    return data
    
def write_meta_yaml(filename, data):
    with open(filename, 'w') as f:
        f.write(yaml.dump(data, default_flow_style=False))

def write_out_csv(events_yaml, bgs_yaml, out_path):
    """Read from events yaml and background yaml file. Create mixed audio yaml
    file. 
    
    Args: 
      events_yaml: string, path of events_yaml. 
      bgs_yaml: string, path of background yaml. 
      out_path: path to write out created yaml file. 
      
    Returns:
      None. 
    """
    events_data = read_meta_yaml(events_yaml)
    bgs_data = read_meta_yaml(bgs_yaml)
    clip_sec = cfg.clip_sec
    create_folder(os.path.dirname(out_path))
    
    rs = np.random.RandomState(0)
    rs.shuffle(bgs_data)
    
    out_yaml = []
    cnt = 0
    for event_key in events_data.keys():
        for i1 in xrange(len(events_data[event_key])):
            event_na = events_data[event_key][i1]['audio_filename']
            [event_bgn_sec, event_fin_sec] = events_data[event_key][i1]['segment']
            bg_na = bgs_data[cnt]['filepath']
            bg_type = bgs_data[cnt]['classname']
            bg_bgn_sec = rs.randint(low=0, high=25)
            bg_fin_sec = bg_bgn_sec + clip_sec
            
            out_yaml.append({'event_name': event_na, 
                             'event_type': event_key, 
                             'event_segment': [event_bgn_sec, event_fin_sec], 
                             'bg_name': bg_na, 
                             'bg_type': bg_type, 
                             'bg_segment': [bg_bgn_sec, bg_fin_sec]})
            cnt += 1
    print(out_path)
    write_meta_yaml(out_path, out_yaml)
    

def main():
    """Create yaml file containing mixed information for training and testing. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--workspace', type=str)
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    tr_events_yaml = os.path.join(dataset_dir, "cv_setup/events_devtrain.yaml")
    te_events_yaml = os.path.join(dataset_dir, "cv_setup/events_devtest.yaml")
    tr_bgs_yaml = os.path.join(dataset_dir, "cv_setup/bgs_devtrain.yaml")
    te_bgs_yaml = os.path.join(dataset_dir, "cv_setup/bgs_devtest.yaml")
    
    write_out_csv(events_yaml=tr_events_yaml, 
                  bgs_yaml=tr_bgs_yaml, 
                  out_path=os.path.join(workspace, "mixed_yaml", "training.csv"))
    write_out_csv(events_yaml=te_events_yaml, 
                  bgs_yaml=te_bgs_yaml, 
                  out_path=os.path.join(workspace, "mixed_yaml", "testing.csv"))
    
    
if __name__ == '__main__':
    main()