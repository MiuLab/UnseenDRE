# export BERT_BASE_DIR="/home/media1129/Documents/dialogre/bert/BERT/uncased_L-12_H-768_A-12"
# export BERT_BASE_DIR="/home/media1129/Documents/ZZZ_Dialogre_Baseline/BERT/uncased_L-12_H-768_A-12"
export BERT_BASE_DIR="/home/media1129/Documents/ZZZ_Dia_setA/BERT/uncased_L-12_H-768_A-12"
# ZZZ_Dia_setA

# Change class_3_balance_on_A/dev.json to test.json then can get predict on testset
# true_dev.json(now)=dev.json dev.json(now)=test.json

CUDA_VISIBLE_DEVICES="2" \
python predict_self.py \
--epoch_num 0 \
--task_name bert \
--do_eval \
--data_dir ./class_3_balance_on_A \
--vocab_file $BERT_BASE_DIR/vocab.txt \
--bert_config_file $BERT_BASE_DIR/bert_config.json \
--max_seq_length 512 \
--output_dir train_on_A \
--eval_batch_size 128 \
--gradient_accumulation_steps 2

CUDA_VISIBLE_DEVICES="2" \
python predict_self.py \
--epoch_num 1 \
--task_name bert \
--do_eval \
--data_dir ./class_3_balance_on_A \
--vocab_file $BERT_BASE_DIR/vocab.txt \
--bert_config_file $BERT_BASE_DIR/bert_config.json \
--max_seq_length 512 \
--output_dir train_on_A \
--eval_batch_size 128 \
--gradient_accumulation_steps 2

CUDA_VISIBLE_DEVICES="2" \
python predict_self.py \
--epoch_num 2 \
--task_name bert \
--do_eval \
--data_dir ./class_3_balance_on_A \
--vocab_file $BERT_BASE_DIR/vocab.txt \
--bert_config_file $BERT_BASE_DIR/bert_config.json \
--max_seq_length 512 \
--output_dir train_on_A \
--eval_batch_size 128 \
--gradient_accumulation_steps 2

