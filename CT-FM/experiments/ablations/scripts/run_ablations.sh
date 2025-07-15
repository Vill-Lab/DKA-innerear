export CUDA_VISIBLE_DEVICES=0,1,3
lighter fit --config=./experiments/ablations/base.yaml,./experiments/ablations/frameworks/simclr.yaml,./experiments/ablations/backbones/segresenc.yaml
lighter fit --config=./experiments/ablations/base.yaml,./experiments/ablations/frameworks/simclr_intrasample.yaml,./experiments/ablations/backbones/segresenc.yaml

lighter fit --config=./experiments/ablations/base.yaml,./experiments/ablations/frameworks/vicreg.yaml,./experiments/ablations/backbones/segresenc.yaml
lighter fit --config=./experiments/ablations/base.yaml,./experiments/ablations/frameworks/vicreg_intrasample.yaml,./experiments/ablations/backbones/segresenc.yaml

lighter fit --config=./experiments/ablations/base.yaml,./experiments/ablations/frameworks/simsiam.yaml,./experiments/ablations/backbones/segresenc.yaml
lighter fit --config=./experiments/ablations/base.yaml,./experiments/ablations/frameworks/simsiam_intrasample.yaml,./experiments/ablations/backbones/segresenc.yaml
