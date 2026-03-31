# ELLab ML Study
- dataset: PathMNIST / CIFAR10
- task: image classification
- pytorch
- from scratch
    - pytorch 공식 문서 / tutorial
    - AI x
- 코드 모듈화
    - 재사용이 가능하고
    - 새로운 모듈 추가가 쉽고
    - 다양한 세팅을 코드 수정 없이 실행 가능하도록
    - 학습 loss, epoch별 test accuracy -> tensorboard로 기록하도록

# Quick Start
```bash
uv sync
uv run python main.py --model resnet-20
uv run python main.py --model resnet-20 --dataset_name pathmnist

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
