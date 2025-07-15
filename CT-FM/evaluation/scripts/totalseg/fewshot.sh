## README
## The name of the experiment, project and wandb tag are set in the vars section as CLI overrides. 
## These parameters organize how to label the wandb upload and where the model checkpoints are saved. 

export WANDB_ENTITY=aim-harvard
ct_fm_path="/mnt/data1/CT_FM/latest_fm_checkpoints/original/epoch=449-step=225000-v1.ckpt"
# ######################### Few-shot train samples ############################
# 20-shot 
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/segresnetds_random.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=2 --vars#name="baseline" --vars#project="totalseg" --vars#wandb_group='few-shot_20' 
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/unet_suprem.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=2 --vars#name="suprem_unet" --vars#project="totalseg" --vars#wandb_group='few-shot_20'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/segresnetds_ctfm.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=2 --vars#name="ct_fm" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_group='few-shot_20'

# 10-shot
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/segresnetds_random.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=1 --vars#name="baseline" --vars#project="totalseg" --vars#wandb_group='few-shot_10'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/unet_suprem.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=1 --vars#name="suprem_unet" --vars#project="totalseg" --vars#wandb_group='few-shot_10'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/segresnetds_ctfm.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=1 --vars#name="ct_fm" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_group='few-shot_10'

# 5-shot
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/segresnetds.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=0.5 --vars#name="baseline" --vars#project="totalseg" --vars#wandb_group='few-shot_5'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/unet_suprem.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=0.5 --vars#name="suprem_unet" --vars#project="totalseg" --vars#wandb_group='few-shot_5'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/segresnetds_ctfm.yaml --trainer#callbacks#0#until_epoch=0 --vars#percentage=0.5 --vars#name="ct_fm" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_group='few-shot_5'
