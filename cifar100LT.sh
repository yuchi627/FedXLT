###### ============= Non-IID ==================
## ---- Cifar100 non-IID IF100 -----
python main.py \
--num_classes=10 \
--num_clients=10 \
--num_online_clients=10 \
--num_rounds=100 \
--local_epoch=10 \
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
--imb_factor=0.01 \
--client_classifier=tree \
--cls_ser2cli \
--long_tail \
--num_features=128 \
--server_resampler \
--log_probabilities \
--freeze_epoch=30 \
--lamda_cls_num=3 \
--lamda_data_num=1 \
--lamda_data_rare=3 \
--weight_decay=0.01 \
--dataset=cifar100 \
--thres_many=100 \
--thres_few=20 \
--save_ckpt \
--depth=7 \
--named=cifar100_IF100_res50_ClsC_Re10S_Cls3Data1Rare3Lamdagkt_dep7_preDec001_Fre30_bs128_LT_10C10_100r \
2>&1 | tee output_txt/cifar100_IF100_res50_ClsC_Re10S_Cls3Data1Rare3Lamdagkt_dep7_preDec001_Fre30_bs128_LT_10C10_100r.txt


## ---- Cifar100 non-IID IF50 -----
python main.py \
--num_classes=10 \
--num_clients=10 \
--num_online_clients=10 \
--num_rounds=100 \
--local_epoch=10 \
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
--imb_factor=0.05 \
--client_classifier=tree \
--cls_ser2cli \
--long_tail \
--num_features=128 \
--server_resampler \
--log_probabilities \
--freeze_epoch=30 \
--lamda_cls_num=3 \
--lamda_data_num=1 \
--lamda_data_rare=3 \
--weight_decay=0.01 \
--dataset=cifar100 \
--thres_many=100 \
--thres_few=20 \
--save_ckpt \
--depth=7 \
--named=cifar100_IF50_res50_ClsC_Re10S_Cls3Data1Rare3Lamdagkt_dep7_preDec001_Fre30_bs128_LT_10C10_100r \
2>&1 | tee output_txt/cifar100_IF50_res50_ClsC_Re10S_Cls3Data1Rare3Lamdagkt_dep7_preDec001_Fre30_bs128_LT_10C10_100r.txt


## ---- Cifar100 non-IID IF10 -----
python main.py \
--num_classes=10 \
--num_clients=10 \
--num_online_clients=10 \
--num_rounds=100 \
--local_epoch=10 \
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
--long_tail \
--num_features=128 \
--server_resampler \
--log_probabilities \
--freeze_epoch=30 \
--lamda_cls_num=3 \
--lamda_data_num=1 \
--lamda_data_rare=3 \
--weight_decay=0.01 \
--dataset=cifar100 \
--thres_many=100 \
--thres_few=20 \
--save_ckpt \
--depth=7 \
--named=cifar100_IF10_res50_ClsC_Re10S_Cls3Data1Rare3Lamdagkt_dep7_preDec001_Fre30_bs128_LT_10C10_100r \
2>&1 | tee output_txt/cifar100_IF10_res50_ClsC_Re10S_Cls3Data1Rare3Lamdagkt_dep7_preDec001_Fre30_bs128_LT_10C10_100r.txt


###### ============= IID ==================
## ---- Cifar100 IID IF100 -----
python main.py \
--num_classes=10 \
--num_clients=10 \
--num_online_clients=10 \
--num_rounds=100 \
--local_epoch=10 \
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
--imb_factor=0.01 \
--client_classifier=tree \
--cls_ser2cli \
--long_tail \
--num_features=128 \
--server_resampler \
--log_probabilities \
--freeze_epoch=30 \
--lamda_cls_num=3 \
--lamda_data_num=1 \
--lamda_data_rare=3 \
--weight_decay=0.01 \
--dataset=cifar100 \
--thres_many=100 \
--thres_few=20 \
--save_ckpt \
--depth=7 \
--iid \
--named=cifar100_IF100_res50_ClsC_Re10S_Cls3Data1Rare3Lamdagkt_dep7_preDec001_Fre30_bs128_LT_10C10_100r_IID \
2>&1 | tee output_txt/cifar100_IF100_res50_ClsC_Re10S_Cls3Data1Rare3Lamdagkt_dep7_preDec001_Fre30_bs128_LT_10C10_100r_IID.txt


## ---- Cifar100 IID IF50 -----
python main.py \
--num_classes=10 \
--num_clients=10 \
--num_online_clients=10 \
--num_rounds=100 \
--local_epoch=10 \
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
--imb_factor=0.05 \
--client_classifier=tree \
--cls_ser2cli \
--long_tail \
--num_features=128 \
--server_resampler \
--log_probabilities \
--freeze_epoch=30 \
--lamda_cls_num=3 \
--lamda_data_num=1 \
--lamda_data_rare=3 \
--weight_decay=0.01 \
--dataset=cifar100 \
--thres_many=100 \
--thres_few=20 \
--save_ckpt \
--depth=7 \
--iid \
--named=cifar100_IF50_res50_ClsC_Re10S_Cls3Data1Rare3Lamdagkt_dep7_preDec001_Fre30_bs128_LT_10C10_100r_IID \
2>&1 | tee output_txt/cifar100_IF50_res50_ClsC_Re10S_Cls3Data1Rare3Lamdagkt_dep7_preDec001_Fre30_bs128_LT_10C10_100r_IID.txt


## ---- Cifar100 IID IF10 -----
python main.py \
--num_classes=10 \
--num_clients=10 \
--num_online_clients=10 \
--num_rounds=100 \
--local_epoch=10 \
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
--long_tail \
--num_features=128 \
--server_resampler \
--log_probabilities \
--freeze_epoch=30 \
--lamda_cls_num=3 \
--lamda_data_num=1 \
--lamda_data_rare=3 \
--weight_decay=0.01 \
--dataset=cifar100 \
--thres_many=100 \
--thres_few=20 \
--save_ckpt \
--depth=7 \
--iid \
--named=cifar100_IF10_res50_ClsC_Re10S_Cls3Data1Rare3Lamdagkt_dep7_preDec001_Fre30_bs128_LT_10C10_100r_IID \
2>&1 | tee output_txt/cifar100_IF10_res50_ClsC_Re10S_Cls3Data1Rare3Lamdagkt_dep7_preDec001_Fre30_bs128_LT_10C10_100r_IID.txt
