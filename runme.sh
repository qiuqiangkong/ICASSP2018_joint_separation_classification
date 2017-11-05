# You need to modify your dataset path. 
DATASET_DIR="/vol/vssp/AP_datasets/audio/dcase2017/task2/TUT-rare-sound-events-2017-development/data/source_data"

WORKSPACE=pwd

# Create yaml file containing information of mixed event with background. 
python create_split_data_csv.py --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Create mixed audio. 
python prepare_data.py mix --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Calculate log Mel feature for all audio clips. 
python prepare_data.py calculate_logmel --workspace=$WORKSPACE

# Pack feature files to a single hdf5 file. 
python prepare_data.py pack_features --workspace=$WORKSPACE

# Compute scaler for features. 
python prepare_data.py compute_scaler --workspace=$WORKSPACE

# Train model. 
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main.py train --workspace=$WORKSPACE --cla_mapping=global_rank_pooling

# Recognize. 
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main.py recognize --workspace=$WORKSPACE --model_name=md10000_iters.p

# Get stats from recognized probabilites. 
python main.py get_stats --workspace=$WORKSPACE

# Plot segmentation masks. 
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main.py plot_seg_masks --workspace=$WORKSPACE --model_name=md10000_iters.p

# Do source separation
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python separate.py jsc_separation --workspace=$WORKSPACE --model_name=md10000_iters.p

# Evaluate separation SDR, SIR, SAR
python evaluate_separation.py --workspace=$WORKSPACE --sep_type=jsc_separation
