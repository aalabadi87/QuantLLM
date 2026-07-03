"""
QuantLLM v2.0 - Ultra-fast LLM Quantization & GGUF Export

The simplest way to load, quantize, fine-tune, and export LLMs.

Features:
    - Load any HuggingFace model with automatic quantization
    - Export to GGUF, ONNX, MLX formats with proper quantization
    - Fine-tune with LoRA
    - Push to HuggingFace Hub with auto-generated model cards

Example:
    >>> from quantllm import turbo
    >>> 
    >>> # Load any model (auto-quantizes to 4-bit)
    >>> model = turbo("meta-llama/Llama-3.2-3B")
    >>> 
    >>> # Generate text
    >>> model.generate("Hello, world!")
    >>> 
    >>> # Export to GGUF with Q4_K_M quantization
    >>> model.export("gguf", "model.Q4_K_M.gguf", quantization="Q4_K_M")
    >>> 
    >>> # Push to HuggingFace Hub
    >>> model.push("username/my-model", format="gguf", quantization="Q4_K_M")
"""

import importlib
import os
import sys
from typing import Any, Dict

__version__ = "2.0.0"
__title__ = "QuantLLM"
__description__ = "Ultra-fast LLM Quantization & Export (GGUF, ONNX, MLX)"
__author__ = "Dark Coder"

_lazy_module_map: Dict[str, str] = {
    # Core API
    "HardwareProfiler": "quantllm.core",
    "SmartConfig": "quantllm.core",
    "ModelAnalyzer": "quantllm.core",
    "TurboModel": "quantllm.core",
    "turbo": "quantllm.core",

    # Compilation
    "compile_model": "quantllm.core",
    "compile_for_inference": "quantllm.core",
    "compile_for_training": "quantllm.core",
    "compile_for_max_speed": "quantllm.core",
    "is_compile_supported": "quantllm.core",
    "CompiledModelWrapper": "quantllm.core",

    # Flash Attention
    "flash_attention": "quantllm.core",
    "is_flash_attention_available": "quantllm.core",
    "enable_flash_attention_for_model": "quantllm.core",
    "FlashAttentionWrapper": "quantllm.core",

    # Memory Optimization
    "MemoryManager": "quantllm.core",
    "DynamicOffloader": "quantllm.core",
    "GradientCheckpointManager": "quantllm.core",
    "CPUOffloadOptimizer": "quantllm.core",
    "setup_memory_efficient_training": "quantllm.core",

    # Training
    "AutoBatchSizeFinder": "quantllm.core",
    "LoRAAutoConfig": "quantllm.core",
    "TrainingConfig": "quantllm.core",
    "TrainingCallbacks": "quantllm.core",
    "auto_configure_training": "quantllm.core",
    "load_training_data": "quantllm.core",

    # Export
    "UniversalExporter": "quantllm.core",
    "ExportFormat": "quantllm.core",
    "export_model": "quantllm.core",

    # GGUF Export & Quantization
    "convert_to_gguf": "quantllm.quant",
    "quantize_gguf": "quantllm.quant",
    "export_to_gguf": "quantllm.quant",
    "check_llama_cpp": "quantllm.quant",
    "install_llama_cpp": "quantllm.quant",
    "ensure_llama_cpp_installed": "quantllm.quant",
    "GGUF_QUANT_TYPES": "quantllm.quant",
    "QUANT_RECOMMENDATIONS": "quantllm.quant",

    # Hub
    "QuantLLMHubManager": "quantllm.hub",
    "ModelCardGenerator": "quantllm.hub",
    "generate_model_card": "quantllm.hub",

    # Utils
    "configure_logging": "quantllm.utils",
    "enable_logging": "quantllm.utils",
    "MemoryTracker": "quantllm.utils",
    "QuantLLMProgress": "quantllm.utils",
    "print_header": "quantllm.utils",
    "print_success": "quantllm.utils",
    "print_error": "quantllm.utils",
    "print_info": "quantllm.utils",
    "print_warning": "quantllm.utils",
    "print_banner": "quantllm.utils",
    "is_banner_shown": "quantllm.utils",
    "reset_banner": "quantllm.utils",
    "console": "quantllm.utils",
    "QUANTLLM_ORANGE": "quantllm.utils",
}

__all__ = [
    # Main API
    "turbo",
    "TurboModel",
    "SmartConfig",
    "HardwareProfiler",
    "ModelAnalyzer",

    # GGUF Export & Quantization
    "convert_to_gguf",
    "quantize_gguf",
    "export_to_gguf",
    "check_llama_cpp",
    "install_llama_cpp",
    "ensure_llama_cpp_installed",
    "GGUF_QUANT_TYPES",
    "QUANT_RECOMMENDATIONS",

    # Hub
    "QuantLLMHubManager",
    "ModelCardGenerator",
    "generate_model_card",

    # Utils
    "configure_logging",
    "enable_logging",
    "MemoryTracker",
    "QuantLLMProgress",
    "print_header",
    "print_success",
    "print_error",
    "print_info",
    "print_warning",
    "print_banner",
    "show_banner",
    "is_banner_shown",
    "reset_banner",
    "console",
    "QUANTLLM_ORANGE",
]


def _import_lazy_attribute(name: str) -> Any:
    module_name = _lazy_module_map.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module = importlib.import_module(module_name)
    return getattr(module, name)


def __getattr__(name: str) -> Any:
    if name in _lazy_module_map:
        return _import_lazy_attribute(name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)


def _try_configure_logging(level: str = "WARNING") -> None:
    try:
        configure_logging = _import_lazy_attribute("configure_logging")
        configure_logging(level)
    except Exception:
        pass


def _should_show_banner() -> bool:
    """Determine if banner should be shown on import."""
    env_banner = os.environ.get("QUANTLLM_BANNER", "").lower()
    if env_banner in {"0", "false"}:
        return False
    if not sys.stdout.isatty():
        return env_banner in {"1", "true"}
    return True


def show_banner(force: bool = False) -> None:
    """Display the QuantLLM banner."""
    if force or _should_show_banner():
        try:
            print_banner = _import_lazy_attribute("print_banner")
            print_banner(__version__, force=force)
        except Exception:
            pass


_try_configure_logging("WARNING")

if _should_show_banner():
    try:
        print_banner = _import_lazy_attribute("print_banner")
        print_banner(__version__)
    except Exception:
        pass
