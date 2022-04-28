cd .. && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train \
--in_model_dir "pretrained_model/chinese-roberta-wwm-ext" \
--out_model_dir "model/ctc" \
--epochs "50" \
--batch_size "168" \
--max_seq_len "128" \
--learning_rate "5e-4" \
--train_fp "data/example.txt" \
--test_fp "data/example.txt" \
--random_seed_num "22" \
--check_val_every_n_epoch "1" \
--early_stop_times "20" \
--warmup_steps "-1" \
--dev_data_ratio "0.1" \
--training_mode "normal" \
--amp true \
--freeze_embedding false

