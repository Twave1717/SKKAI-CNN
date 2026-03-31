import argparse
import os

import torch
from PIL import Image

import main


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, inputs, target_class=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        score = logits[:, target_class].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze(0).squeeze(0)
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        return logits.detach(), cam.cpu(), target_class


def get_target_layer(model_name, model):
    architecture_name = model_name.split("-")[0]
    if architecture_name == "resnet":
        return model.stage3[-1].plain_sequence[3]
    if architecture_name == "preactresnet":
        return model.stage3[-1].plain_sequence[5]
    if architecture_name == "densenet":
        return model.stage3.block[-1].conv[2]
    raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")


def denormalize_image(image_tensor):
    image = image_tensor.detach().cpu().clone()
    mean = torch.tensor([0.5] * image.shape[0]).view(-1, 1, 1)
    std = torch.tensor([0.5] * image.shape[0]).view(-1, 1, 1)
    image = image * std + mean
    return image.clamp(0.0, 1.0)


def cam_to_rgb(cam):
    red = torch.clamp(1.5 * cam - 0.5, 0.0, 1.0)
    green = torch.clamp(1.0 - torch.abs(2.0 * cam - 1.0), 0.0, 1.0)
    blue = torch.clamp(1.5 * (1.0 - cam) - 0.5, 0.0, 1.0)
    return torch.stack([red, green, blue], dim=0)


def to_pil_image(image_tensor):
    image = image_tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(image)


def save_gradcam_images(image_tensor, cam_tensor, output_prefix):
    original = denormalize_image(image_tensor)
    heatmap = cam_to_rgb(cam_tensor)
    overlay = (0.55 * original + 0.45 * heatmap).clamp(0.0, 1.0)

    original_image = to_pil_image(original)
    heatmap_image = to_pil_image(heatmap)
    overlay_image = to_pil_image(overlay)

    original_image.save(f"{output_prefix}-original.png")
    heatmap_image.save(f"{output_prefix}-heatmap.png")
    overlay_image.save(f"{output_prefix}-overlay.png")


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="pathmnist")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--target_class", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="gradcam")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    dataloaders, metadata = main.get_dataloaders(
        batch_size=1,
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        num_workers=args.num_workers,
        device=device,
    )
    dataloader = dataloaders[args.split]
    if dataloader is None:
        raise ValueError(f"Split '{args.split}' is unavailable for dataset {args.dataset_name}")

    model = main.build_model(args.model, metadata["num_classes"], device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    target_layer = get_target_layer(args.model, model)
    gradcam = GradCAM(model, target_layer)

    dataset = dataloader.dataset
    if args.index < 0 or args.index >= len(dataset):
        gradcam.remove()
        raise IndexError(f"Sample index {args.index} is out of range for split {args.split}")

    image, label = dataset[args.index]
    inputs = image.unsqueeze(0).to(device)
    true_label = int(label)

    with torch.enable_grad():
        logits, cam, target_class = gradcam(inputs, args.target_class)
    predicted_class = logits.argmax(dim=1).item()

    os.makedirs(args.output_dir, exist_ok=True)
    output_prefix = os.path.join(
        args.output_dir,
        f"{args.model}-{args.split}-idx{args.index}-true{true_label}-pred{predicted_class}-target{target_class}",
    )
    save_gradcam_images(inputs[0], cam, output_prefix)
    gradcam.remove()

    label_map = metadata["labels"]
    print(f"Saved Grad-CAM images to {args.output_dir}")
    print(f"True label: {true_label} ({label_map[str(true_label)]})")
    print(f"Predicted label: {predicted_class} ({label_map[str(predicted_class)]})")
    print(f"Target class for Grad-CAM: {target_class} ({label_map[str(target_class)]})")


if __name__ == "__main__":
    main_cli()
