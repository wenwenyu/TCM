#!/bin/bash
#export WORKSPACE=/apdcephfs/share_887471/interns/v_fisherwyu/code/spotter
export WORKSPACE=/apdcephfs/private_v_fisherwyu/code/spotter

#date
datename=$(date +%Y%m%d_%H%M%S)
CONFIG=${WORKSPACE}/configs/BAText/Pretrain/v1_clip_attn_R_50.yaml
WORK_DIR=/apdcephfs/share_887471/interns/v_fisherwyu/model_output/spotter
OUT_DIR=${WORK_DIR}/pretrain/v1_clip_attn_R_50_$datename
GPUS=8
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
#python ${WORKSPACE}/train_net.py --config-file $CONFIG --num-gpus $GPUS OUTPUT_DIR $OUT_DIR
python ${WORKSPACE}/tools/train_net_abc.py --config-file $CONFIG --num-gpus $GPUS OUTPUT_DIR $OUT_DIR \
SOLVER.IMS_PER_BATCH $GPUS CLIP.PIX_CLS_LOSS_WEIGHT 1.0  \
MODEL.WEIGHTS /apdcephfs/private_v_fisherwyu/code/spotter/weights/abc/R-50.pkl

#DATALOADER.NUM_WORKERS 2