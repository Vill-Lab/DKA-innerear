## README
## The name of the experiment, project and wandb tag are set in the vars section as CLI overrides. 
## These parameters organize how to label the wandb upload and where the model checkpoints are saved. 
export WANDB_ENTITY=aim-harvard
ct_fm_path="/mnt/data16/ibro/IDC_SSL_CT/runs/fm/checkpoints/CT_FM_SimCLR_SegResNetDS/epoch=449-step=225000-v1.ckpt"

lighter fit --config=./evaluation/msd.yaml,./evaluation/overrides/msd_task06_lung.yaml,./evaluation/baselines/segresnetds.yaml --vars#name="baseline"
lighter fit --config=./evaluation/msd.yaml,./evaluation/overrides/msd_task06_lung.yaml  --system#model#trunk#ckpt_path=$ct_fm_path --vars#name="ct_fm"
