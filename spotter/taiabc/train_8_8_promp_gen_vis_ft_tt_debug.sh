export WORKSPACE=/apdcephfs/private_v_fisherwyu/code/spotter
#export WORKSPACE=/apdcephfs/share_887471/interns/v_fisherwyu/code/spotter

CONFIG=${WORKSPACE}/configs/TESTR/TotalText/TESTR_R_50_Polygon.yaml
WORK_DIR=/apdcephfs/share_887471/interns/v_fisherwyu/model_output/spotter
OUT_DIR=${WORK_DIR}/totaltext/TESTR_R_50_Polygon
GPUS=1
PORT=5555

cp -r /apdcephfs/private_v_fisherwyu/code/spotter/adet /workspace/spotter/
cp -r /apdcephfs/private_v_fisherwyu/code/spotter/projects /workspace/spotter/
cd /workspace/spotter
rm -rf AdelaiDet.egg-info
rm -rf build
#cd ${WORKSPACE}
python setup.py build develop
cd ${WORKSPACE}
#python ${WORKSPACE}/train_net.py --config-file $CONFIG --num-gpus $GPUS OUTPUT_DIR $OUT_DIR
python ${WORKSPACE}/tools/train_net.py --config-file $CONFIG --num-gpus $GPUS OUTPUT_DIR $OUT_DIR SOLVER.IMS_PER_BATCH 1