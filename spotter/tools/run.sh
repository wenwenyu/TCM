OMP_NUM_THREADS=1 python tools/train_net.py --config-file configs/BAText/ICDAR2015/v1_attn_R_50_debug.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 1
OMP_NUM_THREADS=1 python tools/train_net.py --config-file configs/BAText/ICDAR2015/v1_clip_attn_R_50_debug.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 2
OMP_NUM_THREADS=1 python tools/train_net.py --config-file configs/BAText/TotalText/v1_clip_attn_R_50_debug.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 1




#testr
# tt->tt
python tools/train_net.py --config-file configs/TESTR/TotalText/TESTR_R_50_Polygon_debug.yaml --eval-only MODEL.WEIGHTS weights/TESTR/totaltext_testr_R_50_polygon.pth

# tt->ic15
python tools/train_net.py --config-file configs/TESTR/ICDAR15/TESTR_R_50_Polygon_debug.yaml --eval-only MODEL.WEIGHTS weights/TESTR/totaltext_testr_R_50_polygon.pth
# det. 0.8517,0.7824,0.8156

# tt->ctw
python tools/train_net.py --config-file configs/TESTR/CTW1500/TESTR_R_50_Polygon_debug.yaml --eval-only MODEL.WEIGHTS weights/TESTR/totaltext_testr_R_50_polygon.pth
# 0.4273,0.5197,0.4690

# ic15->ctw
python tools/train_net.py --config-file configs/TESTR/CTW1500/TESTR_R_50_Polygon_debug.yaml --eval-only MODEL.WEIGHTS weights/TESTR/icdar15_testr_R_50_polygon.pth
# 0.4056,0.4390,0.4217

# ctw->ic15
python tools/train_net.py --config-file configs/TESTR/ICDAR15/TESTR_R_50_Polygon_debug.yaml --eval-only MODEL.WEIGHTS weights/TESTR/ctw1500_testr_R_50_polygon.pth
# 0.6115,0.2205,0.3241

# ic15->tt
python tools/train_net.py --config-file configs/TESTR/TotalText/TESTR_R_50_Polygon_debug.yaml --eval-only MODEL.WEIGHTS weights/TESTR/icdar15_testr_R_50_polygon.pth
#0.8881,0.7209,0.7958

# ctw->tt
python tools/train_net.py --config-file configs/TESTR/TotalText/TESTR_R_50_Polygon_debug.yaml --eval-only MODEL.WEIGHTS weights/TESTR/ctw1500_testr_R_50_polygon.pth
# 0.6568,0.3613,0.4662

#testr-clip
# ic15->ctw
python tools/train_net.py --config-file configs/TESTR/CTW1500/TESTR_CLIP_R_50_Polygon_debug.yaml --eval-only MODEL.WEIGHTS weights/TESTR-CLIP/ic15/model_0018399.pth


#abc
# tt->ic15
python tools/train_net_abc.py --config-file configs/BAText/ICDAR15/v1_attn_R_50.yaml --eval-only MODEL.WEIGHTS weights/abc/tt/tt_e2e_attn_R_50.pth
#0.8948,0.7699,0.8276

# tt->ctw
python tools/train_net_abc.py --config-file configs/BAText/CTW1500/v1_attn_R_50.yaml --eval-only MODEL.WEIGHTS weights/abc/tt/tt_e2e_attn_R_50.pth
#0.4139,0.4929,0.4500

# ic15->ctw
python tools/train_net_abc.py --config-file configs/BAText/CTW1500/v1_attn_R_50.yaml --eval-only MODEL.WEIGHTS weights/abc/ic15/v1_ic15_finetuned.pth
# 0.4069,0.4144,0.4106

# ic15->tt
python tools/train_net_abc.py --config-file configs/BAText/TotalText/v1_attn_R_50.yaml --eval-only MODEL.WEIGHTS weights/abc/ic15/v1_ic15_finetuned.pth
# 0.8698,0.6762,0.7609

# ctw->ic15
python tools/train_net_abc.py --config-file configs/BAText/ICDAR15/v1_attn_R_50.yaml --eval-only MODEL.WEIGHTS weights/abc/ctw/ctw1500_attn_R_50.pth
#0.6748,0.3727,0.4801

# ctw->tt
python tools/train_net_abc.py --config-file configs/BAText/TotalText/v1_attn_R_50.yaml --eval-only MODEL.WEIGHTS weights/abc/ctw/ctw1500_attn_R_50.pth
# 0.6360,0.3211,0.4268