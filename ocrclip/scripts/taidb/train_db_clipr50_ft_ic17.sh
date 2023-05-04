#export WORKSPACE=/apdcephfs/share_887471/common/ocr_benchmark/fisherwwyu/OCRCLIP/ocrclip
#export WORKSPACE=/apdcephfs/private_v_fisherwyu/code/OCRCLIP/ocrclip
export WORKSPACE=/apdcephfs/share_887471/interns/v_willwhua/wenwenyu/code/detcp/ocrclip

#CONFIG=${WORKSPACE}/configs/textdet/dbnet/clip_dbnet_r50_fpnc_prompt_20e_8x24_st_real3_pretrain_taiji.py
#CONFIG=${WORKSPACE}/configs/textdet/dbnet/dbnet_clipr50_fpnc_1200e_8x16_ic17_ft_ranger_taiji.py
CONFIG=${WORKSPACE}/configs/textdet/dbnet/dbnet_clipr50_fpnc_1200e_8x16_ic17_ft_ranger_taiji_hua.py
#WORK_DIR=/apdcephfs/share_887471/interns/v_fisherwyu/model_output/clip_saved/detclip
WORK_DIR=/apdcephfs/share_887471/interns/v_willwhua/wenwenyu/model_output/clip_saved/detclip
GPUS=8
PORT=5555

if [ ${GPUS} == 1 ]; then
    python ${WORKSPACE}/train.py  $CONFIG --work-dir=${WORK_DIR}
else
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        ${WORKSPACE}/train.py $CONFIG --work-dir=${WORK_DIR} --launcher pytorch
fi
