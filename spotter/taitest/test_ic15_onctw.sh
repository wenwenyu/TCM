#!/bin/bash
export WORKSPACE=/apdcephfs/private_v_fisherwyu/code/spotter
#export WORKSPACE=/apdcephfs/share_887471/interns/v_fisherwyu/code/spotter

#date
CONFIG=${WORKSPACE}/configs/TESTR/CTW1500/TESTR_CLIP_R_50_Polygon.yaml

OUT_DIR=/apdcephfs/share_887471/interns/v_fisherwyu/model_output/testr-clip/ctw
CKPTS=/apdcephfs/share_887471/interns/v_fisherwyu/model_output/spotter/icdar15/TESTR_R_50_Polygon_20220426_202722
#DATA_NAME=totaltext
#GPUS=1
#PORT=5555

cp -r ${WORKSPACE}/adet /workspace/spotter/
cp -r ${WORKSPACE}/projects /workspace/spotter/
cd /workspace/spotter
rm -rf AdelaiDet.egg-info
rm -rf build
rm -rf AdelaiDet/_C.*.so
python setup.py build develop
cd ${WORKSPACE}

for file in $(ls $CKPTS)
do
    if [ "${file##*.}" = "pth" ]; then
#        echo $file
        cd ${WORKSPACE}
        echo $CKPTS/${file}
        python ${WORKSPACE}/tools/train_net.py --config-file $CONFIG --eval-only SOLVER.IMS_PER_BATCH 1 \
        OUTPUT_DIR $OUT_DIR MODEL.WEIGHTS $CKPTS/${file}
    fi
done > /apdcephfs/share_887471/interns/v_fisherwyu/model_output/spotter-clip/test_dir/ctw/ic15_onctw_test_ckpts.txt
