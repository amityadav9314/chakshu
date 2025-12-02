import torch
import sys
import platform
from rich.console import Console

console = Console()

def check_gpu():
    console.print("[bold blue]Chakshu GPU Verification[/bold blue]")
    console.print(f"Python: {sys.version.split()[0]}")
    console.print(f"OS: {platform.system()} {platform.release()}")
    
    try:
        console.print(f"PyTorch Version: {torch.__version__}")
        
        if hasattr(torch.version, 'hip') and torch.version.hip:
            console.print(f"[green]ROCm (HIP) Version: {torch.version.hip}[/green]")
        else:
            console.print("[yellow]No ROCm (HIP) version detected in PyTorch.[/yellow]")

        if torch.cuda.is_available():
            console.print(f"[bold green]GPU Available: Yes[/bold green]")
            device_count = torch.cuda.device_count()
            console.print(f"Device Count: {device_count}")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                console.print(f"Device {i}: [cyan]{device_name}[/cyan]")
                
                # Test tensor operation
                try:
                    x = torch.rand(5, 3).to(f'cuda:{i}')
                    console.print(f"  [green]Tensor allocation successful on device {i}[/green]")
                    y = torch.rand(5, 3).to(f'cuda:{i}')
                    z = x + y
                    console.print(f"  [green]Tensor addition successful on device {i}[/green]")
                except Exception as e:
                    console.print(f"  [bold red]Tensor operation failed on device {i}: {e}[/bold red]")
        else:
            console.print("[bold red]GPU Available: No[/bold red]")
            console.print("Note: For AMD GPUs, ensure PyTorch is installed with ROCm support.")
            
    except ImportError:
        console.print("[bold red]PyTorch not installed.[/bold red]")

if __name__ == "__main__":
    check_gpu()
