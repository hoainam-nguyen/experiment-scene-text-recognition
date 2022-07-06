CUDA_VISIBLE_DEVICES=2 \
python3 train.py \
    --exp_name train_MJST_300k_0707 \
    --batch_size 192 \
    --valid_data /mlcv/WorkingSpace/SceneText/namnh/Research-STR-draft/data_lmdb_release/validation \
    --train_data /mlcv/WorkingSpace/SceneText/namnh/Research-STR-draft/data_lmdb_release/training \
    --select_data MJ-ST \
    --batch_ratio 0.5-0.5 \
    --Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn

