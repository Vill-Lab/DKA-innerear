## README
## The name of the experiment, project and wandb tag are set in the vars section as CLI overrides. 
## These parameters organize how to label the wandb upload and where the model checkpoints are saved. 


clear_processes() { pkill -9 -f lighter && pkill -9 -f wandb; }


# Weights paths
ct_fm_path="/mnt/data16/ibro/IDC_SSL_CT/runs/fm/checkpoints/CT_FM_SimCLR_SegResNetDS/epoch=449-step=225000-v1.ckpt"
suprem_path="/home/ibrahim/Projects/CT_FM/baselines/SuPreM_UNet/supervised_suprem_unet_2100.pth"
vista3d_path="/home/ibrahim/Projects/CT_FM/baselines/VISTA3D/vista3d/models/model.pt"

datetime=$(date +'%Y-%m-%d_%H-%M-%S')

export WANDB_ENTITY=aim-harvard
# export CUDA_VISIBLE_DEVICES=2,3


# Fit - SinoCT

lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=100 --trainer#callbacks#0#until_epoch=0 --vars#name="baseline_100pct" --vars#datetime=${datetime} --vars#mode=fit_sinoct && clear_processes
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=100 --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm_100pct" --vars#datetime=${datetime} --vars#mode=fit_sinoct --system#model#trunk#ckpt_path=$ct_fm_path && clear_processes
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=100 --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unetencoder_100pct" --vars#datetime=${datetime} --vars#mode=fit_sinoct --system#model#trunk#ckpt_path=$suprem_path && clear_processes
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=100 --trainer#callbacks#0#until_epoch=0 --vars#name="vista3d_segresencoder_100pct" --vars#datetime=${datetime} --vars#mode=fit_sinoct --system#model#trunk#ckpt_path=$vista3d_path  --trainer#accumulate_grad_batches=2 --system#batch_size=8 && clear_processes

lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=10 --trainer#callbacks#0#until_epoch=0 --vars#name="baseline_10pct" --vars#datetime=${datetime} --vars#mode=fit_sinoct && clear_processes
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=10 --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm_10pct" --vars#datetime=${datetime} --vars#mode=fit_sinoct --system#model#trunk#ckpt_path=$ct_fm_path && clear_processes
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=10 --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unetencoder_10pct" --vars#datetime=${datetime} --vars#mode=fit_sinoct --system#model#trunk#ckpt_path=$suprem_path && clear_processes
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=10 --trainer#callbacks#0#until_epoch=0 --vars#name="vista3d_segresencoder_10pct" --vars#datetime=${datetime} --vars#mode=fit_sinoct --system#model#trunk#ckpt_path=$vista3d_path  --trainer#accumulate_grad_batches=2 --system#batch_size=8 && clear_processes

lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=5 --trainer#callbacks#0#until_epoch=0 --vars#name="baseline_5pct" --vars#datetime=${datetime} --vars#mode=fit_sinoct && clear_processes
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=5 --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm_5pct" --vars#datetime=${datetime} --vars#mode=fit_sinoct --system#model#trunk#ckpt_path=$ct_fm_path && clear_processes
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=5 --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unetencoder_5pct" --vars#datetime=${datetime} --vars#mode=fit_sinoct --system#model#trunk#ckpt_path=$suprem_path && clear_processes
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=5 --trainer#callbacks#0#until_epoch=0 --vars#name="vista3d_segresencoder_5pct" --vars#datetime=${datetime} --vars#mode=fit_sinoct --system#model#trunk#ckpt_path=$vista3d_path  --trainer#accumulate_grad_batches=2 --system#batch_size=8 && clear_processes

lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=2 --trainer#callbacks#0#until_epoch=0 --vars#name="baseline_2pct" --vars#datetime=${datetime} --vars#mode=fit_sinoct && clear_processes
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=2 --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm_2pct" --vars#datetime=${datetime} --vars#mode=fit_sinoct --system#model#trunk#ckpt_path=$ct_fm_path && clear_processes
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=2 --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unetencoder_2pct" --vars#datetime=${datetime} --vars#mode=fit_sinoct --system#model#trunk#ckpt_path=$suprem_path && clear_processes
lighter fit --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=2 --trainer#callbacks#0#until_epoch=0 --vars#name="vista3d_segresencoder_2pct" --vars#datetime=${datetime} --vars#mode=fit_sinoct --system#model#trunk#ckpt_path=$vista3d_path && clear_processes

# Test - SinoCT

lighter test --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=100 --trainer#callbacks#0#until_epoch=0 --vars#name="baseline_100pct" --vars#datetime=${datetime} --vars#mode=test_sinoct --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False && clear_processes
lighter test --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=100 --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm_100pct" --vars#datetime=${datetime} --vars#mode=test_sinoct --system#model#trunk#ckpt_path=$ct_fm_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False && clear_processes
lighter test --config=./evaluation/sinoct.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=100 --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unetencoder_100pct" --vars#datetime=${datetime} --vars#mode=test_sinoct --system#model#trunk#ckpt_path=$suprem_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False && clear_processes
lighter test --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=100 --trainer#callbacks#0#until_epoch=0 --vars#name="vista3d_segresencoder_100pct" --vars#datetime=${datetime} --vars#mode=test_sinoct --system#model#trunk#ckpt_path=$vista3d_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False && clear_processes

lighter test --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=10 --trainer#callbacks#0#until_epoch=0 --vars#name="baseline_10pct" --vars#datetime=${datetime} --vars#mode=test_sinoct --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False && clear_processes
lighter test --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=10 --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm_10pct" --vars#datetime=${datetime} --vars#mode=test_sinoct --system#model#trunk#ckpt_path=$ct_fm_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False && clear_processes
lighter test --config=./evaluation/sinoct.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=10 --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unetencoder_10pct" --vars#datetime=${datetime} --vars#mode=test_sinoct --system#model#trunk#ckpt_path=$suprem_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False && clear_processes
lighter test --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=10 --trainer#callbacks#0#until_epoch=0 --vars#name="vista3d_segresencoder_10pct" --vars#datetime=${datetime} --vars#mode=test_sinoct --system#model#trunk#ckpt_path=$vista3d_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False && clear_processes

lighter test --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=5 --trainer#callbacks#0#until_epoch=0 --vars#name="baseline_5pct" --vars#datetime=${datetime} --vars#mode=test_sinoct --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False && clear_processes
lighter test --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=5 --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm_5pct" --vars#datetime=${datetime} --vars#mode=test_sinoct --system#model#trunk#ckpt_path=$ct_fm_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False && clear_processes
lighter test --config=./evaluation/sinoct.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=5 --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unetencoder_5pct" --vars#datetime=${datetime} --vars#mode=test_sinoct --system#model#trunk#ckpt_path=$suprem_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False && clear_processes
lighter test --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=5 --trainer#callbacks#0#until_epoch=0 --vars#name="vista3d_segresencoder_5pct" --vars#datetime=${datetime} --vars#mode=test_sinoct --system#model#trunk#ckpt_path=$vista3d_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False && clear_processes

lighter test --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=2 --trainer#callbacks#0#until_epoch=0 --vars#name="baseline_2pct" --vars#datetime=${datetime} --vars#mode=test_sinoct --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False && clear_processes
lighter test --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=2 --trainer#callbacks#0#until_epoch=0 --vars#name="ct_fm_2pct" --vars#datetime=${datetime} --vars#mode=test_sinoct --system#model#trunk#ckpt_path=$ct_fm_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False && clear_processes
lighter test --config=./evaluation/sinoct.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=2 --trainer#callbacks#0#until_epoch=0 --vars#name="suprem_unetencoder_2pct" --vars#datetime=${datetime} --vars#mode=test_sinoct --system#model#trunk#ckpt_path=$suprem_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False && clear_processes
lighter test --config=./evaluation/sinoct.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=2 --trainer#callbacks#0#until_epoch=0 --vars#name="vista3d_segresencoder_2pct" --vars#datetime=${datetime} --vars#mode=test_sinoct --system#model#trunk#ckpt_path=$vista3d_path --trainer#devices=1 --trainer#strategy="auto" --trainer#sync_batchnorm=False && clear_processes


# Test - CQ500

lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=100 --vars#name="baseline_100pct" --vars#datetime=${datetime} --vars#mode=test_cq500 && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=100 --vars#name="ct_fm_100pct" --vars#datetime=${datetime} --vars#mode=test_cq500 --system#model#trunk#ckpt_path=$ct_fm_path && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=100 --vars#name="suprem_unetencoder_100pct" --vars#datetime=${datetime} --vars#mode=test_cq500 --system#model#trunk#ckpt_path=$suprem_path && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=100 --vars#name="vista3d_segresencoder_100pct" --vars#datetime=${datetime} --vars#mode=test_cq500 --system#model#trunk#ckpt_path=$vista3d_path && clear_processes

lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=10 --vars#name="baseline_10pct" --vars#datetime=${datetime} --vars#mode=test_cq500 && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=10 --vars#name="ct_fm_10pct" --vars#datetime=${datetime} --vars#mode=test_cq500 --system#model#trunk#ckpt_path=$ct_fm_path && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=10 --vars#name="suprem_unetencoder_10pct" --vars#datetime=${datetime} --vars#mode=test_cq500 --system#model#trunk#ckpt_path=$suprem_path && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=10 --vars#name="vista3d_segresencoder_10pct" --vars#datetime=${datetime} --vars#mode=test_cq500 --system#model#trunk#ckpt_path=$vista3d_path && clear_processes

lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=5 --vars#name="baseline_5pct" --vars#datetime=${datetime} --vars#mode=test_cq500 && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=100 --vars#name="baseline_100pct" --vars#datetime=${datetime} --vars#mode=test_cq500 && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=100 --vars#name="ct_fm_100pct" --vars#datetime=${datetime} --vars#mode=test_cq500 --system#model#trunk#ckpt_path=$ct_fm_path && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=100 --vars#name="suprem_unetencoder_100pct" --vars#datetime=${datetime} --vars#mode=test_cq500 --system#model#trunk#ckpt_path=$suprem_path && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=100 --vars#name="vista3d_segresencoder_100pct" --vars#datetime=${datetime} --vars#mode=test_cq500 --system#model#trunk#ckpt_path=$vista3d_path && clear_processes

lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=10 --vars#name="baseline_10pct" --vars#datetime=${datetime} --vars#mode=test_cq500 && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=10 --vars#name="ct_fm_10pct" --vars#datetime=${datetime} --vars#mode=test_cq500 --system#model#trunk#ckpt_path=$ct_fm_path && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=10 --vars#name="suprem_unetencoder_10pct" --vars#datetime=${datetime} --vars#mode=test_cq500 --system#model#trunk#ckpt_path=$suprem_path && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=10 --vars#name="vista3d_segresencoder_10pct" --vars#datetime=${datetime} --vars#mode=test_cq500 --system#model#trunk#ckpt_path=$vista3d_path && clear_processes

lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=5 --vars#name="baseline_5pct" --vars#datetime=${datetime} --vars#mode=test_cq500 && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=5 --vars#name="ct_fm_5pct" --vars#datetime=${datetime} --vars#mode=test_cq500 --system#model#trunk#ckpt_path=$ct_fm_path && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=5 --vars#name="suprem_unetencoder_5pct" --vars#datetime=${datetime} --vars#mode=test_cq500 --system#model#trunk#ckpt_path=$suprem_path && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=5 --vars#name="vista3d_segresencoder_5pct" --vars#datetime=${datetime} --vars#mode=test_cq500 && clear_processes

lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_random.yaml --vars#percentage=2 --vars#name="baseline_2pct" --vars#datetime=${datetime} --vars#mode=test_cq500 && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_ctfm.yaml --vars#percentage=2 --vars#name="ct_fm_2pct" --vars#datetime=${datetime} --vars#mode=test_cq500 --system#model#trunk#ckpt_path=$ct_fm_path && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/unetencoder_suprem.yaml --vars#percentage=2 --vars#name="suprem_unetencoder_2pct" --vars#datetime=${datetime} --vars#mode=test_cq500 --system#model#trunk#ckpt_path=$suprem_path && clear_processes
lighter test --config=./evaluation/cq500.yaml,./evaluation/baselines/segresencoder_vista3d.yaml --vars#percentage=2 --vars#name="vista3d_segresencoder_2pct" --vars#datetime=${datetime} --vars#mode=test_cq500 && clear_processes