"""
gpu_memtest.py
==============
GPU 显存自检：把空闲显存按 1 GiB 分块写满已知的 int64 序列，读回两遍逐元素比对。
RTX PRO 6000 默认关闭 ECC，显存坏区不会报错，只会悄悄改写数据（2026-11-14 的 AT 就是这样崩的，HANDOFF 第 8 节）。
换卡或开机后、跑实验之前先执行一次（约 1 分钟）；GPU 上不能有别的任务。退出码 0 = 通过，1 = 有坏块。

Usage:
    python tools/gpu_memtest.py
"""

import sys

import torch

GIB = 1 << 30
N = GIB // 8          # 每块的 int64 个数
MULT = 2654435761     # 每个元素、每一块的值都不同
HEADROOM_GIB = 6      # 留给比对时的临时张量


def pattern(c: int) -> torch.Tensor:
    return torch.arange(N, device="cuda", dtype=torch.int64) * MULT + c


def main() -> int:
    free, total = torch.cuda.mem_get_info()
    print(f"{torch.cuda.get_device_name(0)}: free {free / GIB:.1f} / {total / GIB:.1f} GiB", flush=True)
    chunks = []
    for c in range(int(free / GIB) - HEADROOM_GIB):
        t = torch.empty(N, dtype=torch.int64, device="cuda")
        t.copy_(pattern(c))
        chunks.append(t)
    torch.cuda.synchronize()
    bad = {}
    for _ in range(2):
        for c, t in enumerate(chunks):
            n_bad = int((t != pattern(c)).sum())
            if n_bad:
                bad[c] = max(bad.get(c, 0), n_bad)
    torch.cuda.synchronize()
    print(f"checked {len(chunks)} GiB twice")
    if not bad:
        print("PASS: no mismatches")
        return 0
    print(f"FAIL: {len(bad)} bad GiB chunks (chunk: mismatched elements) {bad}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
