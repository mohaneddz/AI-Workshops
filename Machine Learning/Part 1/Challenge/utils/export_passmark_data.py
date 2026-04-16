"""
Export PassMark CPU and GPU benchmark data to CSV for use in the preprocessing notebook.
Uses scraper.py (must be in the same folder) to fetch data from PassMark sites.
Run from project root: python export_passmark_data.py
Writes: data/cpu_passmark.csv, data/gpu_passmark.csv
"""
import os
import re
import sys

# Allow importing scraper from same directory (project root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from utils.scraper import Scraper


def _parse_num(val):
    """Parse PassMark numeric field (may be string like '1,234' or number)."""
    if val is None or val == "NA":
        return None
    if isinstance(val, (int, float)):
        return float(val) if not (val != val) else None  # NaN check
    s = re.sub(r"[^0-9.]", "", str(val))
    return float(s) if s else None


def main():
    # Project root = folder containing this script (and data/)
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "data")
    os.makedirs(data_dir, exist_ok=True)

    # CPU: cpubenchmark.net -> name, cpumark, single_thread (thread), cores, threads (logicals)
    print("Fetching CPU benchmarks (cpubenchmark.net)...")
    try:
        cpu_scraper = Scraper(domain="www.cpubenchmark.net")
        cpu_rows = []
        for item in cpu_scraper.items:
            name = item.get("name") or ""
            cpumark = _parse_num(item.get("cpumark"))
            single_thread = _parse_num(item.get("thread"))   # PassMark "thread" = single-thread score
            cores = _parse_num(item.get("cores"))
            logicals = _parse_num(item.get("logicals"))       # thread count (logicals)
            thread_count = logicals if logicals is not None and logicals > 0 else cores
            if name and cpumark is not None:
                cpu_rows.append({
                    "name": name,
                    "cpumark": cpumark,
                    "single_thread": single_thread if single_thread is not None else "",
                    "cores": int(cores) if cores is not None else "",
                    "threads": int(thread_count) if thread_count is not None else "",
                })
        cpu_df = pd.DataFrame(cpu_rows)
        cpu_path = os.path.join(data_dir, "cpu_passmark.csv")
        cpu_df.to_csv(cpu_path, index=False)
        print(f"  Saved {len(cpu_df)} CPUs to {cpu_path}")
    except Exception as e:
        print(f"  CPU fetch failed: {e}")

    # GPU: videocardbenchmark.net -> name, g2d, g3d (use g3d as primary for workload)
    print("Fetching GPU benchmarks (videocardbenchmark.net)...")
    try:
        gpu_scraper = Scraper(domain="www.videocardbenchmark.net")
        gpu_rows = []
        for item in gpu_scraper.items:
            name = item.get("name") or ""
            g2d = _parse_num(item.get("g2d"))
            g3d = _parse_num(item.get("g3d"))
            if name and (g2d is not None or g3d is not None):
                gpu_rows.append({
                    "name": name,
                    "g2d": g2d if g2d is not None else "",
                    "g3d": g3d if g3d is not None else "",
                })
        gpu_df = pd.DataFrame(gpu_rows)
        gpu_path = os.path.join(data_dir, "gpu_passmark.csv")
        gpu_df.to_csv(gpu_path, index=False)
        print(f"  Saved {len(gpu_df)} GPUs to {gpu_path}")
    except Exception as e:
        print(f"  GPU fetch failed: {e}")

    print("Done. Use data/cpu_passmark.csv and data/gpu_passmark.csv in the notebook.")


if __name__ == "__main__":
    main()
