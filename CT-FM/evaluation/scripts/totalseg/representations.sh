## README
## The name of the experiment, project and wandb tag are set in the vars section as CLI overrides. 
## These parameters organize how to label the wandb upload and where the model checkpoints are saved. 

export WANDB_ENTITY=aim-harvard
ct_fm_path="/mnt/data1/CT_FM/latest_fm_checkpoints/original/epoch=449-step=225000-v1.ckpt"

# ######################### Decoder-only ############################
# lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/unet_suprem.yaml --trainer#strategy=ddp_find_unused_parameters_true --vars#name="suprem_unet" --vars#project="totalseg" --vars#wandb_group='representations'
# pkill -9 -f lighter
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/segresnetds_ctfm.yaml --trainer#strategy=ddp_find_unused_parameters_true --vars#name="ct_fm" --vars#project="totalseg" --vars#wandb_group='representations'
