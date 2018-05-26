python train_bilstm_attn_adversarial.py \
--test_train '/home/raghavendra/BackUP/MyWorks/workspace/BMNER_CRF_SSVM_DeepNL_code/finaldata/clef2013Dataset/train/dev_all.features'  \
-T '/home/raghavendra/BackUP/MyWorks/workspace/BMNER_CRF_SSVM_DeepNL_code/finaldata/semeval2013_drugner/train_clef2013/train_all.features' \
-d '/home/raghavendra/BackUP/MyWorks/workspace/BMNER_CRF_SSVM_DeepNL_code/finaldata/semeval2013_drugner/train_clef2013/train_all.features' \
-t '/home/raghavendra/BackUP/MyWorks/workspace/BMNER_CRF_SSVM_DeepNL_code/finaldata/semeval2013_drugner/test_clef2013/test_all.features' \
--score score.test.3features \
--test_crfFeatures_dir '/home/raghavendra/BackUP/MyWorks/workspace/BMNER_CRF_SSVM_DeepNL_code/finaldata/semeval2013_drugner/test_clef2013/features' \
--output_crfFeatures_dir /home/raghavendra/BackUP/MyWorks/workspace/BMNER_CRF_SSVM_DeepNL_code/finaldata/semeval2013_drugner/test_clef2013/features_OP/pytorch_$1.features  \
--clef_eval_script '/home/raghavendra/BackUP/MyWorks/workspace/Hitachi2016Eclipse/bmner/bmner_clef2013_code/NER-pytorch-master/commandline_scripts/semeval2013/apr172018_semeval2013_ner_bilstm_newattn_adversarial.eval.sh' \
--clef_eval_script_arg $1 \
--pre_emb '/home/raghavendra/BackUP/MyWorks/workspace/BMNER_CRF_SSVM_DeepNL_code/finaldata/clef201314_hitachi201416_i2b22010_thyme_semeval201713_docs.txt_cleaned_glove_100d.txt' \
--gazetterfile '/home/raghavendra/BackUP/MyWorks/workspace/BMNER_CRF_SSVM_DeepNL_code/finaldata/semeval2013_drugner/train_clef2013/train.gazetter' \
--pretrained_embedding_size 100 \
--lstm_output_size 100 \
--lr 0.001 \
--nb_epoch 150 \
--lower 0 \
--gazetter \
--include_chars \
--attn_model 'tanh' \
--pos #\
#--caps \
#--suf \
#--pre #\

#--perturb 1 

