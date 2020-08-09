CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --synthetic_train_data_dir ./data/output/ ./data/SynthText_lmdb/ \
  --test_data_dir ./data/IIIT5k/ \
  --batch_size 128 \
  --workers 8 \
  --height 64 \
  --width 256 \
  --voc_type ALLCASES_SYMBOLS \
  --arch ResNet_ASTER \
  --with_lstm \
  --logs_dir logs_wo_rec/baseline/ \
  --real_logs_dir ./data/data/logs/ \
  --max_len 100 \
  --STN_ON \
  --tps_inputsize 32 64 \
  --tps_outputsize 32 100 \
  --tps_margins 0.05 0.05 \
  --stn_activation none \
  --num_control_points 20 \
  --encoder_block 4 \
  --decoder_block 4 \
  --lr 0.00002 \
  # --resume ./logs_wo_rec/baseline/checkpoint.pth.tar \
  
