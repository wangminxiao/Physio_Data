#!/usr/bin/env python3
"""
Check server resources: CPU, memory, disk, GPU, and running jobs.
Run before starting heavy pipeline stages.

Usage: python workzone/check_resources.py
"""
import os
import json
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "workzone" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def check_cpu():
    print("=== CPU ===")
    n_cores = os.cpu_count()
    print(f"  Cores: {n_cores}")

    # Load average
    try:
        load1, load5, load15 = os.getloadavg()
        print(f"  Load avg: {load1:.1f} / {load5:.1f} / {load15:.1f} (1/5/15 min)")
        print(f"  Usage: ~{load1/n_cores*100:.0f}% ({load1:.0f}/{n_cores} cores busy)")
    except OSError:
        load1, load5, load15 = None, None, None

    return {"cores": n_cores, "load_1m": load1, "load_5m": load5, "load_15m": load15}


def check_memory():
    print("\n=== Memory ===")
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        mem = {}
        for line in lines:
            parts = line.split()
            if parts[0] in ("MemTotal:", "MemAvailable:", "MemFree:", "SwapTotal:", "SwapFree:"):
                mem[parts[0].rstrip(":")] = int(parts[1]) / 1024 / 1024  # KB -> GB

        total = mem.get("MemTotal", 0)
        available = mem.get("MemAvailable", 0)
        used = total - available
        print(f"  Total: {total:.1f} GB")
        print(f"  Available: {available:.1f} GB")
        print(f"  Used: {used:.1f} GB ({used/total*100:.0f}%)")
        swap = mem.get("SwapTotal", 0)
        if swap > 0:
            swap_free = mem.get("SwapFree", 0)
            print(f"  Swap: {swap - swap_free:.1f} / {swap:.1f} GB")

        return {"total_gb": round(total, 1), "available_gb": round(available, 1),
                "used_pct": round(used/total*100, 1)}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"error": str(e)}


def check_disk():
    print("\n=== Disk ===")
    paths_to_check = [
        "/",
        "/opt/localdata100tb",
        "/labs/hulab",
        "/tmp",
    ]

    disks = {}
    seen_devices = set()
    for path in paths_to_check:
        if not os.path.exists(path):
            continue
        try:
            usage = shutil.disk_usage(path)
            total_gb = usage.total / 1e9
            free_gb = usage.free / 1e9
            used_gb = usage.used / 1e9
            # Avoid duplicate mounts
            stat = os.stat(path)
            device_key = stat.st_dev
            if device_key in seen_devices:
                continue
            seen_devices.add(device_key)

            print(f"  {path}: {free_gb:.0f} GB free / {total_gb:.0f} GB total ({used_gb/total_gb*100:.0f}% used)")
            disks[path] = {"total_gb": round(total_gb, 1), "free_gb": round(free_gb, 1),
                           "used_pct": round(used_gb/total_gb*100, 1)}
        except Exception as e:
            print(f"  {path}: ERROR {e}")

    # Check output directory specifically
    import yaml
    try:
        with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
            cfg = yaml.safe_load(f)
        for dataset in ["mimic3", "mcmed"]:
            out_dir = cfg.get(dataset, {}).get("output_dir", "")
            if out_dir:
                try:
                    usage = shutil.disk_usage(out_dir if os.path.exists(out_dir) else os.path.dirname(out_dir))
                    free_gb = usage.free / 1e9
                    print(f"  Output ({dataset}): {out_dir} -> {free_gb:.0f} GB free")
                    disks[f"output_{dataset}"] = {"path": out_dir, "free_gb": round(free_gb, 1)}
                except Exception:
                    pass
    except Exception:
        pass

    return disks


def check_gpu():
    print("\n=== GPU ===")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("  No GPU (nvidia-smi failed)")
            return {"available": False}

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpu = {
                    "name": parts[0],
                    "memory_total_mb": int(parts[1]),
                    "memory_used_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                    "utilization_pct": int(parts[4]),
                }
                gpus.append(gpu)
                print(f"  {gpu['name']}: {gpu['memory_free_mb']}MB free / {gpu['memory_total_mb']}MB, {gpu['utilization_pct']}% util")

        return {"available": True, "count": len(gpus), "gpus": gpus}
    except FileNotFoundError:
        print("  No GPU (nvidia-smi not found)")
        return {"available": False}
    except Exception as e:
        print(f"  GPU check error: {e}")
        return {"available": False, "error": str(e)}


def check_running_jobs():
    print("\n=== Running Jobs (this user) ===")
    try:
        user = os.environ.get("USER", "unknown")
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=10)
        lines = [l for l in result.stdout.split("\n") if user in l and "python" in l.lower()]
        print(f"  Python processes by {user}: {len(lines)}")
        for line in lines[:10]:
            parts = line.split()
            if len(parts) >= 11:
                pid, cpu, mem = parts[1], parts[2], parts[3]
                cmd = " ".join(parts[10:])[:80]
                print(f"    PID {pid}: CPU={cpu}% MEM={mem}% {cmd}")
        return {"user": user, "python_processes": len(lines)}
    except Exception as e:
        return {"error": str(e)}


def check_slurm():
    """Check if we're on a SLURM cluster."""
    print("\n=== Job Scheduler ===")
    # SLURM
    try:
        result = subprocess.run(["squeue", "-u", os.environ.get("USER", "")],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            print(f"  SLURM: {len(lines)-1} jobs in queue")
            for line in lines[:5]:
                print(f"    {line}")
            return {"type": "slurm", "jobs": len(lines)-1}
    except FileNotFoundError:
        pass

    # PBS/Torque
    try:
        result = subprocess.run(["qstat", "-u", os.environ.get("USER", "")],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  PBS/Torque detected")
            return {"type": "pbs"}
    except FileNotFoundError:
        pass

    # LSF
    try:
        result = subprocess.run(["bjobs"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  LSF detected")
            return {"type": "lsf"}
    except FileNotFoundError:
        pass

    print("  No job scheduler detected (running on login/shared node)")
    return {"type": "none"}


def recommend(resources):
    """Suggest --workers count based on resources."""
    print("\n=== Recommendation ===")
    cores = resources["cpu"]["cores"]
    load = resources["cpu"].get("load_1m", 0) or 0
    avail_mem = resources["memory"].get("available_gb", 0)

    free_cores = max(1, int(cores - load))
    # Each wfdb worker uses ~1-2 GB for reading + resampling
    mem_workers = max(1, int(avail_mem / 2))
    recommended = min(free_cores, mem_workers, 16)

    print(f"  Free cores: ~{free_cores}")
    print(f"  Memory allows: ~{mem_workers} workers (at ~2 GB each)")
    print(f"  Recommended --workers: {recommended}")

    # Disk warning
    for key, disk in resources["disk"].items():
        if "output" in key and disk.get("free_gb", 999) < 500:
            print(f"  WARNING: Output disk {disk.get('path', key)} has only {disk['free_gb']:.0f} GB free!")

    return {"recommended_workers": recommended}


def main():
    print(f"Server Resource Check")
    print(f"Host: {os.uname().nodename}")
    print(f"User: {os.environ.get('USER', 'unknown')}\n")

    resources = {
        "host": os.uname().nodename,
        "cpu": check_cpu(),
        "memory": check_memory(),
        "disk": check_disk(),
        "gpu": check_gpu(),
        "jobs": check_running_jobs(),
        "scheduler": check_slurm(),
    }
    resources["recommendation"] = recommend(resources)

    out_path = OUT_DIR / "server_resources.json"
    with open(out_path, "w") as f:
        json.dump(resources, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
