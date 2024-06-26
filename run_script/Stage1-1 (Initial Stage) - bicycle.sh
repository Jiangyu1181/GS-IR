# Stage1 (Initial Stage)
python train.py \
-m /mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/ \
-s /mnt/jy_nas/Dataset/openBuildingBRDF/360_v2/bicycle \
--iterations 30000 \
-i images_4 \
-r 1 \
--eval

# baking
python baking.py \
-m /mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/ \
--checkpoint /mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/chkpnt30000.pth \
--bound 16.0 \
--occlu_res 256 \
--occlusion 0.4

# Stage2 (Decomposition Stage)
python train.py \
-m /mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/ \
-s /mnt/jy_nas/Dataset/openBuildingBRDF/360_v2/bicycle \
--start_checkpoint /mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/chkpnt30000.pth \
--iterations 40000 \
-i images_4 \
-r 1 \
--eval \
--metallic \
--indirect


# Evaluation (Novel View Synthesis)
python render.py \
-m /mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/ \
-s /mnt/jy_nas/Dataset/openBuildingBRDF/360_v2/bicycle \
--checkpoint /mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/chkpnt40000.pth \
-i images_4 \
-r 1 \
--eval \
--skip_train \
--pbr \
--metallic \
--indirect

# Evaluation (Normal)
#python normal_eval.py \
#--gt_dir /mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/ \
#--output_dir /mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/test/ours_None

# Evaluation (Albedo)
#python render.py \
#/mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/ \
#-s /mnt/jy_nas/Dataset/openBuildingBRDF/360_v2/bicycle \
#--checkpoint /mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/chkpnt40000.pth \
#--eval \
#--skip_train \
#--brdf_eval

# Relighting
python relight.py \
-m /mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/ \
-s /mnt/jy_nas/Dataset/openBuildingBRDF/360_v2/bicycle \
--checkpoint /mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/chkpnt40000.pth \
--hdri /mnt/jy_nas/Dataset/EnvLight/high_res/test/test_026.hdr \
--eval \
--metallic \
--gamma

# Relighting Evaluation
#python relight_eval.py \
#--output_dir /mnt/jy_nas/CodeOutput/Building_GSIR_original/bicycle/test/ours_None/relight \
#--gt_dir /mnt/jy_nas/Dataset/openBuildingBRDF/360_v2/bicycle