cd /storage/src
SESSION_DIR="../experiments/_layers1_dim256_embedding256_lr0.0001/2022-12-18_202148"
python test.py \
    ../data/sample_dataset_config.txt \
    --config ../configs/bilstm.yml \
    --model_path ${SESSION_DIR}/checkpoints/01.h5 \
    --session_dir ${SESSION_DIR}