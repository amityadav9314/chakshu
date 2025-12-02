import os
import sys
from ultralytics import YOLO
from rich.console import Console
import torch
import time
from datetime import timedelta

console = Console()

def train():
    """
    Train YOLOv8 model on AMD GPU.
    """
    console.print("[bold blue]Initializing Chakshu Training...[/bold blue]")
    
    # Ensure HSA override is set for RDNA 3
    if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
        console.print("[yellow]Warning: HSA_OVERRIDE_GFX_VERSION not set. Setting to 11.0.0 for RX 7800 XT[/yellow]")
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

    # Verify GPU
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        console.print(f"[green]Using GPU: {device_name}[/green]")
        device = 0
    else:
        console.print("[bold red]No GPU detected! Training will be slow.[/bold red]")
        device = 'cpu'

    # Load model
    console.print("[blue]Loading YOLOv8n model...[/blue]")
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train
    console.print("[bold green]Starting Training...[/bold green]")
    try:
        start_time = time.time()
        results = model.train(
            data='dataset.yaml',
            epochs=10,
            imgsz=640,
            device=device,
            batch=16,
            project='runs/detect',
            name='chakshu_yolov8n',
            exist_ok=True,
            # AMD ROCm specific optimizations if needed, usually handled by PyTorch
        )
        end_time = time.time()
        duration = end_time - start_time
        console.print("[bold green]Training Complete![/bold green]")
        console.print(f"[bold cyan]Total Training Time: {str(timedelta(seconds=int(duration)))}[/bold cyan]")
    except Exception as e:
        console.print(f"[bold red]Training Failed: {e}[/bold red]")
        raise e

if __name__ == "__main__":
    train()
