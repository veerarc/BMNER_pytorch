cd /home/raghavendra/BackUP/MyWorks/workspace/BMNER_CRF_SSVM_DeepNL_code
# #convert test crffeaturesformat  to clef2013 pipedformat
java -cp bin/:lib/*:. edu.iiit.cde.r.crf.FeaturesToClef2013PipedConvertor -convertfeaturestopipe -tagfield 2 -corpusdir finaldata/semeval2013_drugner/test_clef2013/txt -inputfeaturesdir finaldata/semeval2013_drugner/test_clef2013/features_OP/pytorch_$1.features -outputpipeddir finaldata/semeval2013_drugner/test_clef2013/features_OP/pytorch_$1.piped > 2_delete
# #evaluate
perl ./scripts/Task1Eval_clef2013.pl -n pytorch_$1.piped -r 1 -t 1a -input finaldata/semeval2013_drugner/test_clef2013/features_OP/pytorch_$1.piped -gold ./finaldata/semeval2013_drugner/test_clef2013/pipe_txt -a
echo "TEST result:"
cat scripts/outputs/pytorch_$1.piped.1.1a.add

mv scripts/outputs/pytorch_$1.piped.1.1a.add scripts/semeval_drugner_outputs

#cd /home/raghavendra/BackUP/MyWorks/workspace/Hitachi2014Eclipse

#java -cp bin/:lib/* edu.iiit.cde.r.nn.DeepNlConvertor ../BMNER_CRF_SSVM_DeepNL_code/finaldata/clef2013Dataset/test/DS ../BMNER_CRF_SSVM_DeepNL_code/finaldata/clef2013Dataset/test/features_OP/pytorch_$1.features ../BMNER_CRF_SSVM_DeepNL_code/finaldata/clef2013Dataset/test/features_OP/pytorch_$1.piped > 1_delete

#perl scripts/Task1Eval_clef2013.pl -n pytorch_$1.piped2 -r 1 -t 1a   -input  ../BMNER_CRF_SSVM_DeepNL_code/finaldata/clef2013Dataset/test/features_OP/pytorch_$1.piped -gold ../BMNER_CRF_SSVM_DeepNL_code/finaldata/clef2013Dataset/test/DSPIPED
#cat scripts/outputs/pytorch_$1.piped2.1.1a.noadd
