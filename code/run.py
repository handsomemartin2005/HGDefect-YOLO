"""Train, evaluate, and run HGDefect-YOLO."""
from pathlib import Path
import argparse
import torch
from register_modules import register


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['check', 'train', 'val', 'predict'])
    p.add_argument('--model', default=str(Path(__file__).parent / 'models/yolov8n-paper-formulas.yaml'))
    p.add_argument('--data')
    p.add_argument('--source')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--project', default='runs/paper_formula')
    args = p.parse_args()
    if args.mode in ['train', 'val'] and not args.data:
        p.error('--data is required')
    if args.mode in ['val', 'predict'] and Path(args.model).suffix != '.pt':
        p.error('Provide a checkpoint trained with this variant via --model')
    if args.mode == 'predict' and not args.source:
        p.error('--source is required')
    register()
    from ultralytics import YOLO
    model = YOLO(args.model)
    if args.mode == 'check':
        model.model.eval()
        with torch.no_grad():
            predictions = model.model(torch.zeros(1, 3, args.imgsz, args.imgsz))
        print('Prediction tensor:', tuple(predictions[0].shape))
        print('Parameters:', sum(p.numel() for p in model.model.parameters()))
    elif args.mode == 'train':
        model.train(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                    device=args.device, seed=args.seed, workers=args.workers,
                    project=args.project, pretrained=False)
    elif args.mode == 'val':
        model.val(data=args.data, batch=args.batch, imgsz=args.imgsz, device=args.device, workers=args.workers)
    else:
        model.predict(source=args.source, imgsz=args.imgsz, device=args.device, save=True, project=args.project)


if __name__ == '__main__':
    main()
