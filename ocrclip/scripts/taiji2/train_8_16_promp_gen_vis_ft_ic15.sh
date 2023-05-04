export WORKSPACE=/apdcephfs/private_v_fisherwyu/code/OCRCLIP/ocrclip
#CONFIG=${WORKSPACE}/configs/textdet/dbnet/clip_dbnet_r50_fpnc_prompt_20e_8x24_st_real3_pretrain_taiji.py
CONFIG=${WORKSPACE}/configs/textdet/dbnet/clip_db_r50_fpnc_prompt_gen_vis_1200e_ic15_taiji.py
#WORK_DIR=/apdcephfs/share_887471/common/ocr_benchmark/model_output/clip_saved/detclip
WORK_DIR=/apdcephfs/share_887471/interns/v_fisherwyu/model_output/clip_saved/detclip
GPUS=1
PORT=5555

if [ ${GPUS} == 1 ]; then
    python ${WORKSPACE}/train.py  $CONFIG --work-dir=${WORK_DIR}
else
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        ${WORKSPACE}/train.py $CONFIG --work-dir=${WORK_DIR} --launcher pytorch
fi
