# export BERT_BASE_DIR="/home/media1129/Documents/TUCORE-GCN/pre-trained_model/BERT/uncased_L-12_H-768_A-12"
# export BERT_BASE_DIR="/home/media1129/Documents/dialogre/bert/BERT/uncased_L-12_H-768_A-12"
# export BERT_BASE_DIR="/home/media1129/Documents/ZZZ_Dialogre_Baseline/BERT/uncased_L-12_H-768_A-12"
export BERT_BASE_DIR="/home/media1129/Documents/ZZZ_Dia_setA/BERT/uncased_L-12_H-768_A-12"

# python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=$BERT_BASE_DIR/bert_model.ckpt --bert_config_file=$BERT_BASE_DIR/bert_config.json --pytorch_dump_path=$BERT_BASE_DIR/pytorch_model.bin


# in_order_epoch3 now   # pos_weight = torch.ones([2])
CUDA_VISIBLE_DEVICES="3" \
python run_classifier.py \
--task_name bert \
--do_train \
--data_dir ./class_3_balance_on_A \
--vocab_file $BERT_BASE_DIR/vocab.txt \
--bert_config_file $BERT_BASE_DIR/bert_config.json \
--init_checkpoint $BERT_BASE_DIR/pytorch_model.bin \
--max_seq_length 512 \
--train_batch_size 4 \
--learning_rate 3e-5 \
--num_train_epochs 5.0 \
--output_dir train_on_A \
--gradient_accumulation_steps 2
# --do_eval \


















# change data folder in run_classifier.py and torch.tenor([4])
# 0.78->0.59 (5 20class)



# rm bert_f1/model_best.pt && cp -r bert_f1 bert_f1c &&

# rm bert_f1_augment_relation_indomain/model_best.pt
# cp -r bert_f1_augment_relation_indomain bert_f1c_augment_relation_indomain


