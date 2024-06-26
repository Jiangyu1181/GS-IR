# Stage1 (Initial Stage)
python train.py \
-m /mnt/jy_nas/CodeOutput/Building_GSIR_original/00004 \
-s /mnt/jy_nas/Dataset/openBuildingBRDF/00004 \
--iterations 30000 \
--eval

# baking
python baking.py \
-m /mnt/jy_nas/CodeOutput/Building_GSIR_original/00004/ \
--checkpoint /mnt/jy_nas/CodeOutput/Building_GSIR_original/00004/chkpnt30000.pth \
--bound 1.5 \
--occlu_res 128 \
--occlusion 0.25

# Stage2 (Decomposition Stage)
python train.py \
-m /mnt/jy_nas/CodeOutput/Building_GSIR_original/00004/ \
-s /mnt/jy_nas/Dataset/openBuildingBRDF/00004 \
--start_checkpoint /mnt/jy_nas/CodeOutput/Building_GSIR_original/00004/chkpnt30000.pth \
--iterations 40000 \
--eval \
--gamma \
--metallic \
--indirect

# Evaluation (Novel View Synthesis)
python render.py \
-m /mnt/jy_nas/CodeOutput/Building_GSIR_original/00004/ \
-s /mnt/jy_nas/Dataset/openBuildingBRDF/00004 \
--checkpoint /mnt/jy_nas/CodeOutput/Building_GSIR_original/00004/chkpnt40000.pth \
--eval \
--skip_train \
--pbr \
--gamma \
--indirect \
--metallic

# Evaluation (Normal)
#python normal_eval.py \
#--gt_dir /mnt/jy_nas/CodeOutput/Building_GSIR_original/00004/ \
#--output_dir /mnt/jy_nas/CodeOutput/Building_GSIR_original/00004/test/ours_None

# Evaluation (Albedo)
#python render.py \
#-m /mnt/jy_nas/CodeOutput/Building_GSIR_original/00004/ \
#-s /mnt/jy_nas/Dataset/openBuildingBRDF/00004 \
#--checkpoint /mnt/jy_nas/CodeOutput/Building_GSIR_original/00004/chkpnt40000.pth \
#--eval \
#--skip_train \
#--brdf_eval

# Relighting
python relight.py \
-m /mnt/jy_nas/CodeOutput/Building_GSIR_original/00004/ \
-s /mnt/jy_nas/Dataset/openBuildingBRDF/00004 \
--checkpoint /mnt/jy_nas/CodeOutput/Building_GSIR_original/00004/chkpnt40000.pth \
--hdri /mnt/jy_nas/Dataset/EnvLight/high_res/test/test_026.hdr \
--eval \
--metallic \
--gamma

# Relighting Evaluation
#python relight_eval.py \
#--output_dir /mnt/jy_nas/CodeOutput/Building_GSIR_original/00004/test/ours_None/relight \
#--gt_dir /mnt/jy_nas/Dataset/openBuildingBRDF/00004