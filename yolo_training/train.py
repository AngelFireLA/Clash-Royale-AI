import os
from ultralytics import YOLO
import yaml


def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


def train_yolov11(
        data_yaml='data.yaml',
        model_name='yolov11n.pt',  # This will only be used if not resuming
        epochs=50,
        batch=16,
        device='auto',  # 'cpu', 'cuda:0', etc.
        project='runs/train',
        name='yolov11_gamebot_run1',
        exist_ok=True,
        verbose=True,
        save_period=None,
        resume=False  # Resume from checkpoint: either bool or string checkpoint path
):
    """
    Trains a YOLOv11 model using the Ultralytics API with custom augmentation settings.

    The augmentation settings are modified to disable all augmentations except:
      - Horizontal flip (fliplr)
      - Rotation (degrees)
      - Saturation and brightness adjustments (hsv_s and hsv_v)

    Other augmentations (vertical flip, translate, scale, shear, mosaic, mixup, etc.) are set to 0.

    Parameters:
        data_yaml (str): Path to the data.yaml configuration file.
        model_name (str): Pretrained model to start with.
        epochs (int): Number of training epochs.
        batch (int): Batch size.
        device (str): Device to use for training.
        project (str): Project directory for saving training runs.
        name (str): Specific run name within the project.
        exist_ok (bool): Whether to overwrite existing runs.
        verbose (bool): Whether to print detailed training logs.
        save_period (int): If provided, saves weights every `save_period` epochs.
        resume (bool/str): If True, resumes training from the last checkpoint. If string,
                             it resumes from that specific checkpoint.

    Returns:
        model (YOLO): Trained YOLO model instance.
    """
    # Define augmentation parameters.
    # Only horizontal flip, rotation, saturation/brightness, and blur are enabled.
    aug_args = {
        "flipud": 0.0,  # Disable vertical flip.
        "fliplr": 0.5,  # Enable horizontal flip with probability 0.5.
        "degrees": 10.0,  # Allow rotation up to 10 degrees.
        "hsv_h": 0.0,  # Disable hue alteration.
        "hsv_s": 0.5,  # Enable saturation adjustments.
        "hsv_v": 0.5,  # Enable brightness adjustments.
        "translate": 0.0,  # Disable translation.
        "scale": 0.0,  # Disable scaling.
        "shear": 0.0,  # Disable shearing.
        "perspective": 0.0,  # Disable perspective transform.
        "mosaic": 0.0,  # Disable mosaic augmentation.
        "mixup": 0.0,  # Disable mixup augmentation.
        "copy_paste": 0.0,  # Disable copy-paste augmentation.
        "bgr": 0.0,  # Disable channel swapping.
    }

    print("Loading YOLOv11 model...")
    if resume:
        checkpoint = os.path.join(project, name, 'weights', 'last.pt')
        if isinstance(resume, str):
            checkpoint = resume  # Allow passing a specific checkpoint path.
        print(f"Resuming training from checkpoint: {checkpoint}")
        model = YOLO(checkpoint)  # Load model from checkpoint.
    else:
        model = YOLO(model_name)  # Start new training.

    print("Starting training with custom augmentation...")
    # Prepare training parameters with augmentation overrides.
    train_params = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch,
        "device": device,
        "project": project,
        "name": name,
        "exist_ok": exist_ok,
        "resume": resume,
        "verbose": verbose
    }
    if save_period:
        train_params["save_period"] = save_period

    # Merge augmentation settings into training parameters.
    train_params.update(aug_args)

    results = model.train(**train_params)

    print("Training completed.")
    return model

def export_yolov11(
        model,
        export_formats=['onnx', 'torchscript'],  # Supported: 'onnx', 'torchscript', 'engine'
        dynamic=False,
        simplify=True,
        optimize=True,
        device='auto'
):
    """
    Exports the trained YOLOv11 model to specified formats.

    Parameters:
        model (YOLO): Trained YOLO model instance.
        export_formats (list): List of formats to export ('onnx', 'torchscript', 'engine').
        dynamic (bool): Whether to use dynamic axes for ONNX.
        simplify (bool): Whether to simplify the ONNX model.
        optimize (bool): Whether to optimize the TorchScript model.
        device (str): Device to use for exporting ('cpu', 'cuda:0', etc.).

    Returns:
        None
    """
    for fmt in export_formats:
        print(f"Exporting model to {fmt} format...")
        if fmt == 'onnx':
            model.export(format=fmt, dynamic=dynamic, simplify=simplify)
        elif fmt == 'torchscript':
            model.export(format=fmt, optimize=optimize)
        elif fmt == 'engine':
            model.export(format=fmt)
        else:
            print(f"Unsupported export format: {fmt}")
        print(f"Exported model to {fmt} format successfully.")


def main():
    # Absolute path to data.yaml (update this path as needed)
    data_yaml = 'dataset/data.yaml'
    # Configuration parameters
    model_name = 'yolo11n.pt'  # Starting model
    epochs = 100
    batch = 32
    device = 'cuda:0'  # Use GPU if available
    project = 'runs/train'
    run_name = 'yolov11_gamebot_run11'
    export_formats = ['onnx', 'torchscript']  # You can add 'engine' if needed

    # Check if data.yaml exists.
    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(f"Data configuration file '{data_yaml}' not found.")

    # Uncomment to perform training from scratch or resume training
    model = train_yolov11(
        data_yaml=data_yaml,
        model_name=model_name,
        epochs=epochs,
        batch=batch,
        device=device,
        project=project,
        name=run_name,
        exist_ok=True,
        verbose=True,
        save_period=5,
        resume=False
    )

    # Export the model.
    export_yolov11(
        model=model,
        export_formats=export_formats,
        dynamic=False,
        simplify=True,
        optimize=True,
        device=device
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting...")
