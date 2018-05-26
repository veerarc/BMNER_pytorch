# BMNER_pytorch

Script to Run Bi-LSTM with general attention function:

python train_bilstm_attn_adversarial.py \
-T '/home/raghavendra/BackUP/MyWorks/workspace/BMNER_CRF_SSVM_DeepNL_code/finaldata/clef2013Dataset/train/train_all.features' \
-d '/home/raghavendra/BackUP/MyWorks/workspace/BMNER_CRF_SSVM_DeepNL_code/finaldata/clef2013Dataset/train/dev_all.features' \
-t '/home/raghavendra/BackUP/MyWorks/workspace/BMNER_CRF_SSVM_DeepNL_code/finaldata/clef2013Dataset/test/test.features' \
--score score.test.3features \
--test_crfFeatures_dir '/home/raghavendra/BackUP/MyWorks/workspace/BMNER_CRF_SSVM_DeepNL_code/finaldata/clef2013Dataset/test/features' \
--output_crfFeatures_dir /home/raghavendra/BackUP/MyWorks/workspace/BMNER_CRF_SSVM_DeepNL_code/finaldata/clef2013Dataset/test/features_OP/pytorch_$1.features  \
--clef_eval_script '/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/bmner/bmner_clef2013_code/NER-pytorch-master/commandline_scripts/apr132018_bilstm_newattn_adversarial.eval.sh' \
--pre_emb '/home/raghavendra/BackUP/tools/GloVe-master/deepnlclef_glove_train_test_Vectors_vocab_vectors' \
--pretrained_embedding_size 50 \
--lstm_output_size 100 \
--lr 0.001 \
--nb_epoch 150 \
--lower 0 \
--attn_model 'general'
