#!/bin/bash
export WORKSPACE=/apdcephfs/private_v_fisherwyu/code/spotter
#export WORKSPACE=/apdcephfs/share_887471/interns/v_fisherwyu/code/spotter

#date
datename=$(date +%Y%m%d_%H%M%S)
CONFIG=${WORKSPACE}/configs/BAText/TotalText/v1_clip_attn_R_50.yaml
WORK_DIR=/apdcephfs/share_887471/interns/v_fisherwyu/model_output/abc
OUT_DIR=${WORK_DIR}/totaltext/v1_clip_attn_R_50_$datename
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
OMP_NUM_THREADS=1 python ${WORKSPACE}/tools/train_net_abc.py --config-file $CONFIG --num-gpus $GPUS OUTPUT_DIR $OUT_DIR \
SOLVER.IMS_PER_BATCH 1 CLIP.PIX_CLS_LOSS_WEIGHT 1.0 DATALOADER.NUM_WORKERS 2 \
SOLVER.BASE_LR 0.00001 SOLVER.MAX_ITER 200000 \
TEST.EVAL_PERIOD 200 \
MODEL.WEIGHTS /apdcephfs/private_v_fisherwyu/code/spotter/weights/abc/tt/tt_e2e_attn_R_50.pth