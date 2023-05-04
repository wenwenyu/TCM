#export WORKSPACE=/apdcephfs/share_887471/common/ocr_benchmark/fisherwwyu/OCRCLIP/ocrclip
export WORKSPACE=/apdcephfs/private_v_fisherwyu/code/OCRCLIP/ocrclip
#export NCCL_IB_DISABLE=1
#CONFIG=${WORKSPACE}/configs/textdet/dbnet/clip_dbnet_r50_fpnc_prompt_20e_8x24_st_real3_pretrain_taiji.py
#CONFIG=${WORKSPACE}/configs/textdet/dbnet/clip_db_r50_fpnc_prompt_gen_20e_8x16_st_real3_pretrain_taiji.py
CONFIG=${WORKSPACE}/configs/textdet/dbnet/clip_db_r50_fpnc_prompt_gen_vis_18_20e_8x16_st150k_real3_pretrain_taiji.py
python -m torch.distributed.launch --nproc_per_node=8 --master_port=5555 \
${WORKSPACE}/train_ocrclip.py $CONFIG \
--work-dir=/apdcephfs/share_887471/interns/v_fisherwyu/model_output/clip_saved/detclip --launcher pytorch