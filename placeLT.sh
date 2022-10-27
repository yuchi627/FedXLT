## ---- place WeightedAgg rare3 cls3 data1 Non-IID -----
python main.py \
--num_classes=10 \
--num_clients=10 \
--num_online_clients=10 \
--num_rounds=100 \
--local_epoch=10 \
--batch_size_test=256 \
--batch_size_global_training=256 \
--batch_size_local_training=256 \
--global_epoch=10 \
--lr_net=1e-5 \
--lr=0.001 \
--lr_block=0.001 \
--net=resnet50 \
--milestones=30,60,90,100 \
--workers=16 \
--classifier=tree \
--fed_method=gkt \
--imb_factor=0.1 \
--client_classifier=tree \
--cls_ser2cli \
--num_features=128 \
--server_resampler \
--log_probabilities \
--freeze_epoch=30 \
--lamda_cls_num=3 \
--lamda_data_num=1 \
--lamda_data_rare=3 \
--weight_decay=0.01 \
--dataset=place \
--max_sample_proportion=18 \
--thres_many=100 \
--thres_few=20 \
--save_ckpt \
--depth=9 \
--named=place_res50_ClsC_Re10S_Cls3Data1Rare3Lamdagkt_dep9_preDec001_Fre30_bs128_LT_10C10_100r_samplePro16_NonIID_2 \
2>&1 | tee output_txt/place_res50_ClsC_Re10S_Cls3Data1Rare3Lamdagkt_dep9_preDec001_Fre30_bs128_LT_10C10_100r_samplePro16_NonIID_2.txt


## ---- place WeightedAgg rare3 cls3 data1 IID -----
python main.py \
--num_classes=10 \
--num_clients=10 \
--num_online_clients=10 \
--num_rounds=100 \
--local_epoch=10 \
--batch_size_test=128 \
--batch_size_global_training=128 \
--batch_size_local_training=128 \
--global_epoch=10 \
--lr_net=1e-5 \
--lr=0.001 \
--lr_block=0.001 \
--net=resnet50 \
--milestones=30,60,90,100 \
--workers=16 \
--classifier=tree \
--fed_method=gkt \
--imb_factor=0.1 \
--client_classifier=tree \
--cls_ser2cli \
--num_features=128 \
--server_resampler \
--log_probabilities \
--freeze_epoch=30 \
--lamda_cls_num=3 \
--lamda_data_num=1 \
--lamda_data_rare=3 \
--weight_decay=0.01 \
--dataset=place \
--max_sample_proportion=16 \
--thres_many=100 \
--thres_few=20 \
--save_ckpt \
--depth=9 \
--iid \
--named=place_res50_ClsC_Re10S_Cls3Data1Rare3Lamdagkt_dep9_preDec001_Fre30_bs128_LT_10C10_100r_samplePro16_IID \
2>&1 | tee output_txt/place_res50_ClsC_Re10S_Cls3Data1Rare3Lamdagkt_dep9_preDec001_Fre30_bs128_LT_10C10_100r_samplePro16_IID.txt