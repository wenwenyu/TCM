#!/bin/bash
export WORKSPACE=/apdcephfs/private_v_fisherwyu/code/spotter
#export WORKSPACE=/apdcephfs/share_887471/interns/v_fisherwyu/code/spotter

#date
datename=$(date +%Y%m%d_%H%M%S)
CONFIG=${WORKSPACE}/configs/TESTR/ICDAR15/v1_clip_attn_R_50.yaml
WORK_DIR=/apdcephfs/share_887471/interns/v_fisherwyu/model_output/spotter
OUT_DIR=${WORK_DIR}/icdar15/v1_clip_attn_R_50_$datename
GPUS=1
PORT=5555

#mkdir /workspace
#mkdir /workspace/spotter/
cp -r ${WORKSPACE}/adet /workspace/spotter/
cp -r ${WORKSPACE}/projects /workspace/spotter/
cd /workspace/spotter
rm -rf AdelaiDet.egg-info
rm -rf build
python setup.py build develop
cd ${WORKSPACE}
OMP_NUM_THREADS=1 python ${WORKSPACE}/tools/train_net.py --config-file $CONFIG --num-gpus $GPUS OUTPUT_DIR $OUT_DIR \
SOLVER.IMS_PER_BATCH 1 CLIP.PIX_CLS_LOSS_WEIGHT 1.0 DATALOADER.NUM_WORKERS 1 \
SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 5500 \
TEST.EVAL_PERIOD 2500 \
MODEL.WEIGHTS /apdcephfs/private_v_fisherwyu/code/spotter/weights/abc/ic15/v1_ic15_finetuned.pth