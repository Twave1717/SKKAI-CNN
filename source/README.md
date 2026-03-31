# ML Study
- dataset: PathMNIST / CIFAR10
- task: image classification
- pytorch

# Quick Start
```bash
uv sync
uv run python main.py --model resnet-20
uv run python main.py --model resnet-20 --dataset_name pathmnist
uv run python main.py --model preactresnet-110 --dataset_name pathmnist --early_stopping_patience 20 --early_stopping_metric loss

chmod +x ./tensorboard.sh
./tensorboard.sh
```

# Models
## Resnet
- model name: resnet-n / preactresnet-n
- n: layer size (n = 6a + 2 형태의 값이어야 함)

# Dataset Notes
- `pathmnist`는 `medmnist`의 `train / val / test` split을 그대로 사용합니다.
- 학습 중 평가는 `val` 기준으로 수행하고, 학습 종료 후 `test`를 한 번 더 평가합니다.
- `CIFAR10`도 기존처럼 `--dataset_name CIFAR10`으로 실행할 수 있습니다.
- 기본 early stopping은 `val_loss` 기준이며, best epoch 가중치를 복원한 뒤 최종 `test`를 평가합니다.

# Grad-CAM
```bash
uv run python gradcam.py \
  --model preactresnet-110 \
  --checkpoint checkpoint/preactresnet-110-pathmnist-200-0331_1444.pth \
  --dataset_name pathmnist \
  --split test \
  --index 0
```
- 출력 이미지는 `gradcam/` 아래에 `original / heatmap / overlay` 3장으로 저장됩니다.
