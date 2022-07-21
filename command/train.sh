cd .. && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train \
--in_model_dir "pretrained_model/chinese-roberta-wwm-ext" \
--out_model_dir "model/ctc_train" \
--epochs "50" \
--batch_size "158" \
--max_seq_len "128" \
--learning_rate "5e-5" \
--train_fp "data/comp_data/preliminary_a_data/preliminary_train.json" \
--test_fp "data/comp_data/preliminary_a_data/preliminary_val.json" \
--random_seed_num "42" \
--check_val_every_n_epoch "0.5" \
--early_stop_times "20" \
--warmup_steps "-1" \
--dev_data_ratio "0.01" \
--training_mode "normal" \
--amp true \
--freeze_embedding false

