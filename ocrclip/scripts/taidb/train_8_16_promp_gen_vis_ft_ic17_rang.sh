#export WORKSPACE=/apdcephfs/share_887471/common/ocr_benchmark/fisherwwyu/OCRCLIP/ocrclip
export WORKSPACE=/apdcephfs/private_v_fisherwyu/code/OCRCLIP/ocrclip
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
#CONFIG=${WORKSPACE}/configs/textdet/dbnet/clip_dbnet_r50_fpnc_prompt_20e_8x24_st_real3_pretrain_taiji.py
#CONFIG=${WORKSPACE}/configs/textdet/dbnet/clip_db_r50_fpnc_prompt_gen_vis_1200e_ft_tt_ranger_taiji.py
#CONFIG=${WORKSPACE}/configs/textdet/dbnet/clip_db_r50_fpnc_prompt_gen_vis_1200e_ft_tt_adam_taiji.py
CONFIG=${WORKSPACE}/configs/textdet/dbnet/clip_db_r50_fpnc_prompt_gen_vis_1200e_ft_gen_ic17_ranger_taiji.py
#WORK_DIR=/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip
WORK_DIR=/apdcephfs/share_887471/interns/v_fisherwyu/model_output/clip_saved/detclip
GPUS=8
PORT=5555

if [ ${GPUS} == 1 ]; then
    python ${WORKSPACE}/train.py  $CONFIG --work-dir=${WORK_DIR}
else
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        ${WORKSPACE}/train.py $CONFIG --work-dir=${WORK_DIR} --launcher pytorch
fi
