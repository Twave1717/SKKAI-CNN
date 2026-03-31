import argparse
import os
from datetime import datetime

import architecture
import medmnist
import torch
from medmnist import Evaluator, INFO
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    ToTensor,
)


def build_transforms(dataset_name, num_channels, is_train):
    normalize = Normalize(
        mean=(0.5,) * num_channels,
        std=(0.5,) * num_channels,
    )

    if is_train:
        if dataset_name == "pathmnist":
            return Compose([
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                RandomRotation(15),
                ToTensor(),
                normalize,
            ])
        return Compose([
            RandomHorizontalFlip(),
            RandomResizedCrop(32),
            ToTensor(),
            normalize,
        ])

    return Compose([
        ToTensor(),
        normalize,
    ])


def squeeze_single_label(target):
    return torch.as_tensor(target, dtype=torch.long).view(-1)[0].item()


def build_dataloader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def get_medmnist_dataloaders(batch_size, dataset_name, data_root, num_workers, device):
    flag = dataset_name.lower()
    info = INFO[flag]
    dataset_class = getattr(medmnist, info["python_class"])

    if "3d" in flag:
        raise ValueError(f"Unsupported dataset for 2D CNN architectures: {dataset_name}")
    if info["task"] != "multi-class":
        raise ValueError(f"Unsupported MedMNIST task for CrossEntropyLoss: {info['task']}")

    as_rgb = info["n_channels"] == 1
    num_channels = 3 if as_rgb else info["n_channels"]
    dataset_file = os.path.join(data_root, f"{flag}.npz")

    os.makedirs(data_root, exist_ok=True)

    train_dataset = dataset_class(
        split="train",
        root=data_root,
        download=not os.path.exists(dataset_file),
        as_rgb=as_rgb,
        transform=build_transforms(flag, num_channels, is_train=True),
        target_transform=squeeze_single_label,
    )
    val_dataset = dataset_class(
        split="val",
        root=data_root,
        download=False,
        as_rgb=as_rgb,
        transform=build_transforms(flag, num_channels, is_train=False),
        target_transform=squeeze_single_label,
    )
    test_dataset = dataset_class(
        split="test",
        root=data_root,
        download=False,
        as_rgb=as_rgb,
        transform=build_transforms(flag, num_channels, is_train=False),
        target_transform=squeeze_single_label,
    )

    dataloaders = {
        "train": build_dataloader(train_dataset, batch_size, True, num_workers, device),
        "val": build_dataloader(val_dataset, batch_size, False, num_workers, device),
        "test": build_dataloader(test_dataset, batch_size, False, num_workers, device),
    }
    metadata = {
        "dataset_name": flag,
        "num_classes": len(info["label"]),
        "num_channels": num_channels,
        "labels": info["label"],
        "task": info["task"],
        "evaluator": {
            "val": Evaluator(flag, "val", root=data_root),
            "test": Evaluator(flag, "test", root=data_root),
        },
    }
    return dataloaders, metadata


def get_torchvision_dataloaders(batch_size, dataset_name, data_root, num_workers, device):
    dataset_class = getattr(datasets, dataset_name)
    train_dataset = dataset_class(
        root=data_root,
        train=True,
        download=True,
        transform=build_transforms(dataset_name.lower(), 3, is_train=True),
    )
    test_dataset = dataset_class(
        root=data_root,
        train=False,
        download=True,
        transform=build_transforms(dataset_name.lower(), 3, is_train=False),
    )

    dataloaders = {
        "train": build_dataloader(train_dataset, batch_size, True, num_workers, device),
        "val": None,
        "test": build_dataloader(test_dataset, batch_size, False, num_workers, device),
    }
    metadata = {
        "dataset_name": dataset_name,
        "num_classes": len(train_dataset.classes),
        "num_channels": 3,
        "labels": {str(idx): name for idx, name in enumerate(train_dataset.classes)},
        "task": "multi-class",
        "evaluator": {},
    }
    return dataloaders, metadata


def get_dataloaders(batch_size, dataset_name, data_root, num_workers, device):
    if dataset_name.lower() in INFO:
        return get_medmnist_dataloaders(
            batch_size=batch_size,
            dataset_name=dataset_name,
            data_root=data_root,
            num_workers=num_workers,
            device=device,
        )
    return get_torchvision_dataloaders(
        batch_size=batch_size,
        dataset_name=dataset_name,
        data_root=data_root,
        num_workers=num_workers,
        device=device,
    )


def build_model(model_name, num_classes, device):
    architecture_name, num_layers = model_name.split("-")
    num_layers = int(num_layers)

    if architecture_name == "resnet":
        return architecture.ResNet(num_layers, num_classes).to(device)
    if architecture_name == "preactresnet":
        return architecture.PreActResNet(num_layers, num_classes).to(device)
    if architecture_name == "densenet":
        return architecture.DenseNet(num_layers, num_classes).to(device)
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate(epoch, split_name, dataloader, device, model, loss_fn, tensorboard_writer, evaluator=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0.0
    total_correct = 0
    y_scores = []

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device, dtype=torch.long)

            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            total_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if evaluator is not None:
                y_scores.append(torch.softmax(pred, dim=1).cpu())

    average_loss = total_loss / num_batches
    accuracy = total_correct / size
    print(
        f"{epoch} Epochs {split_name}: \n"
        f" Accuracy: {(accuracy * 100):>0.1f}%, Avg loss: {average_loss:>8f}"
    )

    tensorboard_writer.add_scalar(f"Loss/{split_name}", average_loss, epoch)
    tensorboard_writer.add_scalar(f"Accuracy/{split_name}", accuracy * 100, epoch)

    if evaluator is not None:
        metrics = evaluator.evaluate(torch.cat(y_scores).numpy())
        print(f" {split_name.upper()} AUC: {metrics.AUC:>0.4f}")
        tensorboard_writer.add_scalar(f"AUC/{split_name}", metrics.AUC, epoch)


def train(epochs, train_dataloader, eval_dataloader, eval_name, device, model, loss_fn, optimizer, scheduler, tensorboard_writer, evaluator=None):
    global_step = -1

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}\n-------------------------------")
        model.train()

        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device, dtype=torch.long)

            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 100 == 0:
                print(f"Step [{global_step}/{len(train_dataloader) * epochs}] Loss: {loss.item():.4f}")
                tensorboard_writer.add_scalar("Loss/train", loss.item(), global_step)

        scheduler.step()
        evaluate(
            epoch=epoch,
            split_name=eval_name,
            dataloader=eval_dataloader,
            device=device,
            model=model,
            loss_fn=loss_fn,
            tensorboard_writer=tensorboard_writer,
            evaluator=evaluator,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--model", type=str, default="resnet-20")
    parser.add_argument("--dataset_name", type=str, default="pathmnist")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="checkpoint")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    dataloaders, metadata = get_dataloaders(
        batch_size=args.batch_size,
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        num_workers=args.num_workers,
        device=device,
    )

    sample_X, sample_y = next(iter(dataloaders["train"]))
    print(f"Shape of X [N, C, H, W]: {sample_X.shape}")
    print(f"Shape of y: {sample_y.shape} {sample_y.dtype}")
    print(f"Classes: {metadata['labels']}")

    model = build_model(args.model, metadata["num_classes"], device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.epochs // 2, args.epochs * 3 // 4],
        gamma=0.1,
    )
    tensorboard_writer = SummaryWriter()

    eval_name = "val" if dataloaders["val"] is not None else "test"
    eval_dataloader = dataloaders[eval_name]

    train(
        epochs=args.epochs,
        train_dataloader=dataloaders["train"],
        eval_dataloader=eval_dataloader,
        eval_name=eval_name,
        device=device,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        tensorboard_writer=tensorboard_writer,
        evaluator=metadata["evaluator"].get(eval_name),
    )

    if eval_name != "test":
        print("Final test evaluation\n-------------------------------")
        evaluate(
            epoch=args.epochs,
            split_name="test",
            dataloader=dataloaders["test"],
            device=device,
            model=model,
            loss_fn=loss_fn,
            tensorboard_writer=tensorboard_writer,
            evaluator=metadata["evaluator"].get("test"),
        )

    print("Done!")

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        filename = f"{args.model}-{args.dataset_name}-{args.epochs}-{datetime.now().strftime('%m%d_%H%M')}.pth"
        save_file = os.path.join(args.save_path, filename)
        torch.save(model.state_dict(), save_file)
        print(f"Saved PyTorch Model State to {save_file}")


if __name__ == "__main__":
    main()
