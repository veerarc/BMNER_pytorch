# BMNER_pytorch

Script to Run Bi-LSTM with general attention function:

python train_bilstm_attn_adversarial.py -T 'train/train_all.features' -d 'train/dev_all.features' -t 'test/test.features' --score score.test.3features --test_crfFeatures_dir 'test/features' --output_crfFeatures_dir test/features_OP/pytorch_$1.features  --clef_eval_script 'commandline_scripts/apr132018_bilstm_newattn_adversarial.eval.sh' --pre_emb 'deepnlclef_glove_train_test_Vectors_vocab_vectors' --pretrained_embedding_size 50 --lstm_output_size 100 --lr 0.001 --nb_epoch 150 --lower 0 --attn_model 'general'
