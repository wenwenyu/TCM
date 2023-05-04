export WORKSPACE=/apdcephfs/share_887471/common/ocr_benchmark/fisherwwyu/OCRCLIP/ocrclip
#export WORKSPACE=/apdcephfs/private_mingliangxu/wenwenyu/MASTERv2-main
#export WORKSPACE=/apdcephfs/private_v_ocrlfu/wenwenyu/MASTERv2-main
#python3 -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 \
#--master_addr=127.0.0.1 --master_port=5555 \
#${WORKSPACE}/train.py -c ${WORKSPACE}/configs/xl/config_ranger_flatannelr_h736_w736_mv1_1dq_iam.json -d 0 --local_world_size 1
#python3 -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=5555 ./train_mtp.py -c ./configs/config_mtp_whole_jizhi.json -d 0 --local_world_size 1
#python3 -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 \
#--master_addr=127.0.0.1 --master_port=5555 \
#${WORKSPACE}/train_mtp.py -c ${WORKSPACE}/configs/config_mtp_whole_jizhi.json -d 0 --local_world_size 1
#python3 ${WORKSPACE}/train_mtp.py -c ${WORKSPACE}/configs/config_mtp_whole_jizhi.json -d 0 -dist false
#chmode u+x ${WORKSPACE}/taiji/*.sh
#python3 -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 \
#--master_addr=127.0.0.1 --master_port=5555 \
#${WORKSPACE}/train_ocrclip.py ${WORKSPACE}/configs/textdet/dbnet/clip_dbnet_r50_fpnc_20e_st_real4_pretrain_taiji.py \
#--work-dir=/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip --launcher pytorch
#export NCCL_IB_DISABLE=1
python -m torch.distributed.launch --nproc_per_node=8 --master_port=5555 \
${WORKSPACE}/train_ocrclip.py ${WORKSPACE}/configs/textdet/dbnet/clip_dbnet_r50_fpnc_1200e_8x24_ic15_ft_taiji.py \
--work-dir=/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip --launcher pytorch