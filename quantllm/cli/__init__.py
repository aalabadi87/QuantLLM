"""
QuantLLM CLI - Simple command-line interface

Usage:
    quantllm version     Show version
    quantllm info        Show system info
    quantllm convert     Convert model to GGUF
"""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="quantllm",
        description="QuantLLM - Ultra-fast LLM Quantization",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Version command
    subparsers.add_parser("version", help="Show version")
    
    # Info command
    subparsers.add_parser("info", help="Show system info")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert model to GGUF")
    convert_parser.add_argument("model", help="Model name or path")
    convert_parser.add_argument("-o", "--output", required=True, help="Output GGUF file")
    convert_parser.add_argument("-q", "--quant", default="Q4_K_M", help="Quantization type (default: Q4_K_M)")
    
    # List GGUF command
    list_parser = subparsers.add_parser("list-gguf", help="List GGUF files in a repo or local folder")
    list_parser.add_argument("model", help="HuggingFace repo or local path")
    
    args = parser.parse_args()
    
    if args.command == "version":
        cmd_version()
    elif args.command == "info":
        cmd_info()
    elif args.command == "convert":
        cmd_convert(args.model, args.output, args.quant)
    elif args.command == "list-gguf":
        cmd_list_gguf(args.model)
    else:
        parser.print_help()


def cmd_version():
    """Show version."""
    from quantllm import __version__
    print(f"QuantLLM v{__version__}")


def cmd_info():
    """Show system info."""
    import platform
    print("\n" + "="*50)
    print(" QuantLLM System Info ".center(50, "="))
    print("="*50)
    print(f"\n🐍 Python: {platform.python_version()}")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   CPU: {platform.processor() or 'Unknown'}")

    try:
        import torch
        print(f"\n📦 PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            try:
                print(f"🎮 CUDA: {torch.version.cuda}")
                print(f"   Device: {torch.cuda.get_device_name(0)}")
                mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"   Memory: {mem_gb:.1f} GB")
            except Exception:
                print("🎮 CUDA: Available, but failed to query device info")
        else:
            print("🎮 CUDA: Not available")
    except ImportError:
        print("\n📦 PyTorch: Not installed")
        print("🎮 CUDA: Not available")

    try:
        import flash_attn
        print(f"\n⚡ Flash Attention: {flash_attn.__version__}")
    except ImportError:
        print("\n⚡ Flash Attention: Not installed")

    try:
        import gguf
        print(f"\n🧩 GGUF: {gguf.__version__}")
    except ImportError:
        print("\n🧩 GGUF: Not installed")

    try:
        from transformers import __version__ as transformers_version
        print(f"\n🤗 Transformers: {transformers_version}")
    except ImportError:
        print("\n🤗 Transformers: Not installed")

    print("\n" + "="*50 + "\n")


def cmd_list_gguf(model: str):
    """List available GGUF files in a repository or local folder."""
    print(f"\n🔍 Listing GGUF files for: {model}")
    import os

    try:
        if os.path.isdir(model):
            files = [f for f in os.listdir(model) if f.endswith('.gguf')]
        else:
            from huggingface_hub import list_repo_files
            all_files = list_repo_files(model)
            files = [f for f in all_files if f.endswith('.gguf')]

        if not files:
            print("No GGUF files found.")
            return

        def quant_sort_key(name: str) -> int:
            name_lower = name.lower()
            if 'f32' in name_lower: return 0
            if 'f16' in name_lower: return 1
            if 'q8' in name_lower: return 2
            if 'q6' in name_lower: return 3
            if 'q5_k_m' in name_lower: return 4
            if 'q5_k_s' in name_lower: return 5
            if 'q4_k_m' in name_lower: return 6
            if 'q4_k_s' in name_lower: return 7
            if 'q3_k' in name_lower: return 8
            if 'q2_k' in name_lower: return 9
            return 10

        files = sorted(files, key=quant_sort_key)
        print(f"Found {len(files)} GGUF file(s):")
        for f in files:
            print(f" - {f}")
        print()
    except ImportError:
        print("Error: huggingface_hub is required for listing repository GGUF files. Install with: pip install huggingface-hub")
    except Exception as e:
        print(f"Error listing GGUF files: {e}")


def cmd_convert(model: str, output: str, quant: str):
    """Convert model to GGUF."""
    print(f"\n🚀 Converting {model} to GGUF...")
    
    from quantllm import turbo
    
    # Load model
    print(f"📦 Loading model...")
    m = turbo(model)
    
    # Export
    print(f"📦 Exporting with {quant} quantization...")
    m.export("gguf", output, quantization=quant)
    
    print(f"✅ Done! Created: {output}\n")


if __name__ == "__main__":
    main()