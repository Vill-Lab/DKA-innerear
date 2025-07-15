## README
## The name of the experiment, project and wandb tag are set in the vars section as CLI overrides. 
## These parameters organize how to label the wandb upload and where the model checkpoints are saved. 

export WANDB_ENTITY=aim-harvard
ct_fm_path="/mnt/data1/CT_FM/latest_fm_checkpoints/original/epoch=449-step=225000-v1.ckpt"

# ######################### V2 (VISTA3D split) ############################
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/unet_suprem.yaml --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unet" --vars#project="totalseg" --vars#wandb_group='vista_v2'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/segresnetds_ctfm.yaml --trainer#callbacks#0#until_epoch=0  --vars#name="ct_fm" --vars#project="totalseg" --system#model#trunk#ckpt_path=$ct_fm_path --vars#wandb_group='vista_v2'
lighter fit --config=./evaluation/totalseg.yaml,./evaluation/overrides/totalseg_vista.yaml,./evaluation/baselines/segresnetds_random.yaml --trainer#callbacks#0#until_epoch=0 --vars#name="baseline" --vars#project="totalseg" --vars#wandb_group='vista_v2'
