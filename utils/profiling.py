import csv
import time
import socket
import platform
from pathlib import Path
from functools import wraps
from datetime import datetime

import psutil

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False


# Pfad zur Messdatei
TIMING_FILE = Path("timing_results.csv")


def write_header_if_needed():
    """Erstellt Kopfzeile einmalig."""
    if not TIMING_FILE.exists():
        with TIMING_FILE.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "label",
                "function_name",
                "duration_s",
                "parameters",
                "cpu",
                "ram_total_gb",
                "ram_used_gb",
                "gpu_name",
                "gpu_memory_gb"
            ])


def get_system_info():
    """Systeminformationen für Reproduzierbarkeit."""
    cpu = platform.processor()

    svmem = psutil.virtual_memory()
    ram_total = svmem.total / (1024**3)
    ram_used = svmem.used / (1024**3)

    if GPU_AVAILABLE:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        gpu_name = None
        gpu_mem = None

    return cpu, ram_total, ram_used, gpu_name, gpu_mem


def timing_decorator(label: str = ""):
    """
    Dekoriert eine Funktion mit Zeitmessung + CSV-Ausgabe.
    Wissenschaftlich verwertbar für Masterarbeiten.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            write_header_if_needed()

            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start

            cpu, ram_total, ram_used, gpu_name, gpu_mem = get_system_info()

            # Parameter als repr speichern (für Analyse)
            params = repr(kwargs if kwargs else args)

            # In CSV schreiben
            with TIMING_FILE.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(timespec='seconds'),
                    label,
                    func.__name__,
                    round(duration, 6),
                    params,
                    cpu,
                    round(ram_total, 3),
                    round(ram_used, 3),
                    gpu_name,
                    gpu_mem
                ])

            return result

        return wrapper
    return decorator
