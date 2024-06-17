#!/bin/bash
export WORKSPACE=/apdcephfs/private_v_fisherwyu/code/spotter
#export WORKSPACE=/apdcephfs/share_887471/interns/v_fisherwyu/code/spotter

#date
datename=$(date +%Y%m%d_%H%M%S)
CONFIG=${WORKSPACE}/configs/TESTR/ICDAR15/TESTR_R_50_Polygon.yaml
WORK_DIR=/apdcephfs/share_887471/interns/v_fisherwyu/model_output/spotter
OUT_DIR=${WORK_DIR}/icdar15/TESTR_R_50_Polygon_$datename
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
#python ${WORKSPACE}/train_net.py --config-file $CONFIG --num-gpus $GPUS OUTPUT_DIR $OUT_DIR
python ${WORKSPACE}/tools/train_net.py --config-file $CONFIG --num-gpus $GPUS OUTPUT_DIR $OUT_DIR \
SOLVER.IMS_PER_BATCH 1 CLIP.PIX_CLS_LOSS_WEIGHT 1.0 DATALOADER.NUM_WORKERS 2 SOLVER.OPTIMIZER RANGER \
SOLVER.MAX_ITER 20000 TEST.EVAL_PERIOD 800 SOLVER.BASE_LR 1e-4 SOLVER.LR_BACKBONE 1e-5