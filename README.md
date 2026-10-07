# HGDefect-YOLO

HGDefect-YOLO: Peak-Preserving High-Order Feature Learning for Industrial Surface Defect Detection

Zidu Zhang, Beijing University of Technology
handsomemartin@emails.bjut.edu.cn

## Contents

- code/hgdefect_yolo_modules.py: PPAP-EMA, KDE-guided hypergraph fusion, RGCU and CLAG.
- code/models/: nano-scale model configuration.
- code/configs/: GC10-DET and NEU-DET dataset configuration examples.
- code/register_modules.py: in-memory Ultralytics module registration.
- code/run.py: model construction, training, validation, and prediction.

## Installation

Use Python 3.10 or later and a separate environment. Install a compatible PyTorch/torchvision pair following code/requirements.txt, then run:

```bash
python -m pip install -r code/requirements.txt
```

Ultralytics 8.2.103 is required. The PyTorch version range avoids its incompatibility with the checkpoint-loader default introduced in PyTorch 2.6. Module registration is automatic and does not modify installed source files.

## Usage

Set the dataset root, split paths and class IDs in the chosen dataset YAML. From the repository root:

```bash
python code/run.py check --imgsz 640
python code/run.py train --data code/configs/example_gc10.yaml --device 0 --batch 16 --epochs 300
python code/run.py val --model runs/paper_formula/train/weights/best.pt --data code/configs/example_gc10.yaml --device 0
python code/run.py predict --model runs/paper_formula/train/weights/best.pt --source path/to/image.jpg --device 0
```

The model configuration defaults to 10 classes; training takes the class count from the dataset configuration. Use example_neudet.yaml for NEU-DET. For a standalone six-class construction check, set nc to 6 in the model YAML. Training starts without pretrained weights. Use checkpoints trained with the included architecture for validation and prediction. Datasets and trained weights are not included.

See code/LICENSE for licensing terms. Ultralytics remains a separately installed third-party dependency.
