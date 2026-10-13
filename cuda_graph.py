"""
CUDA Graph 性能对比（无需启动 vLLM）

vLLM 的 CUDA Graph 核心思路：
  1. 用固定地址的 static buffer（replay 时只 copy 新数据，不改 tensor 地址）
  2. warmup 若干次后 capture 一次 graph
  3. 后续相同 shape 的 forward 直接 graph.replay()，减少 CPU 侧 kernel launch 开销

本脚本用随机矩阵模拟 decode 阶段的小 batch MLP，对比 eager vs CUDA Graph。
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn as nn


class DecodeStepMLP(nn.Module):
    """模拟单层 decode：Linear -> SiLU -> Linear -> RMSNorm。"""

    def __init__(self, hidden: int, ffn: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden, ffn, bias=False)
        self.down = nn.Linear(ffn, hidden, bias=False)
        self.norm = nn.RMSNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.down(torch.nn.functional.silu(self.up(x))))


class MultiLayerDecodeBlock(nn.Module):
    """堆叠多层，模拟 vLLM full graph 捕获整段 forward 的场景。"""

    def __init__(self, hidden: int, ffn: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DecodeStepMLP(hidden, ffn) for _ in range(layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _summarize_ms(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    return {
        "mean_ms": statistics.mean(ordered),
        "p50_ms": _percentile(ordered, 50),
        "p99_ms": _percentile(ordered, 99),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


@torch.inference_mode()  # 使用 PyTorch 的 inference_mode，可以提升推理效率并节省内存，禁止 autograd 记录，提高推理场景下的性能
def bench_eager(
    model: nn.Module,                  # 需要 benchmark 的模型，类型为 PyTorch 神经网络模块
    static_input: torch.Tensor,        # 静态输入张量，测试时所有迭代都用同样的数据，减少输入变动对测量的干扰
    static_output: torch.Tensor,       # 静态输出张量，用于保存模型的输出，可复用内存
    iters: int,                        # 测试的正式迭代次数（用于真正计时的循环次数）
    warmup: int,                       # 热身迭代次数，确保 CUDA context 及相关库预热到稳定状态，消除首次运行抖动
) -> dict[str, float]:
    """
    以 eager 执行方式基准（benchmark）一个模型的推理延迟。

    主要流程与原理如下：
    1. 首先利用 warmup 多次推理，促使底层 CUDA 内核生成、cuDNN 选择好合适的算法，提升后续测量稳定性。
    2. 随后对 `iters` 次推理进行准确计时，每轮都将模型输出结果复制到静态输出张量，保证内存可控且能触发真正的计算。
    3. 每次推理后均调用 `_sync(device)`，确保 GPU 所有操作完成并与 CPU 同步，杜绝异步执行导致时间误差。
    4. 最后，将总耗时（毫秒）和平均每轮耗时（毫秒）作为结果返回，便于分析模型推理效率。

    参数说明：
        model: 待测神经网络模型，必须已转到合适的 device 和精度，且设置为 eval()。
        static_input: 用作模型输入的张量，内容固定，仅作用于推理一致性。
        static_output: 存储模型输出的张量，经 copy_ 反复覆盖节省内存。
        iters: 参与正式计时的迭代次数，越大数据越稳定。
        warmup: 热身迭代数，不参与计时。

    返回值：
        一个 dict，包括
          - "total_ms": 总耗时（毫秒）
          - "per_iter_ms": 每次推理平均耗时（毫秒）
    """
    device = static_input.device  # 获取输入张量所在设备（CPU/GPU），用以后续同步

    # Step 1: 热身阶段，执行 warmup 次推理，用于初始化、缓存 CUDA 内核及算法选择
    for _ in range(warmup):
        static_output.copy_(model(static_input))
    _sync(device)  # 强制 GPU 计算同步，确保热身全部完成

    # Step 2: 正式计时推理（iters 次），测量推理总耗时
    start = time.perf_counter()  # 记录开始时间，精度高、适合微基准测试
    for _ in range(iters):
        static_output.copy_(model(static_input))
    _sync(device)  # 确认所有 GPU 工作结束，以保障计时准确
    total_s = time.perf_counter() - start  # 计算推理循环总时长（单位：秒）

    # Step 3: 结果整理与返回
    per_iter_ms = total_s * 1000.0 / iters  # 计算平均单次推理耗时（毫秒）
    return {"total_ms": total_s * 1000.0, "per_iter_ms": per_iter_ms}


@torch.inference_mode()
def bench_cuda_graph(
    model: nn.Module,
    static_input: torch.Tensor,
    static_output: torch.Tensor,
    iters: int,
    warmup: int,
) -> dict[str, float]:
    device = static_input.device
    graph = torch.cuda.CUDAGraph()

    # capture 前 warmup，让 cuDNN 等选好算法
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            static_output.copy_(model(static_input))
    torch.cuda.current_stream().wait_stream(s)
    _sync(device)

    with torch.cuda.graph(graph):
        static_output.copy_(model(static_input))

    # replay 前测 latency 分布
    samples_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        graph.replay()
        _sync(device)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    stats = _summarize_ms(samples_ms)
    stats["total_ms"] = sum(samples_ms)
    stats["per_iter_ms"] = stats["mean_ms"]
    return stats


def build_model(
    hidden: int, ffn: int, layers: int, device: torch.device
) -> nn.Module:
    if layers == 1:
        model: nn.Module = DecodeStepMLP(hidden, ffn)
    else:
        model = MultiLayerDecodeBlock(hidden, ffn, layers)
    return model.to(device=device, dtype=torch.float16).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA Graph vs eager micro-benchmark")
    parser.add_argument("--batch-size", type=int, default=8, help="模拟 decode batch")
    parser.add_argument("--hidden", type=int, default=4096, help="hidden dim")
    parser.add_argument("--ffn", type=int, default=11008, help="FFN intermediate dim")
    parser.add_argument(
        "--layers",
        type=int,
        default=32,
        help="堆叠层数，越大越接近 vLLM full graph 的 kernel 数量",
    )
    parser.add_argument("--iters", type=int, default=500, help="计时迭代次数")
    parser.add_argument("--warmup", type=int, default=20, help="capture 前 warmup 次数")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "当前 PyTorch 未检测到 CUDA。\n"
            "请先安装 GPU 版 PyTorch，例如：\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
            "安装后重新运行本脚本。"
        )

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(
        f"Shape: batch={args.batch_size}, hidden={args.hidden}, ffn={args.ffn}, "
        f"layers={args.layers}, dtype=float16"
    )
    print("-" * 60)

    model = build_model(args.hidden, args.ffn, args.layers, device)

    # vLLM 风格 static buffer：地址固定，replay 时只更新内容
    static_input = torch.randn(
        args.batch_size, args.hidden, device=device, dtype=torch.float16
    )
    static_output = torch.empty_like(static_input)

    eager = bench_eager(model, static_input, static_output, args.iters, args.warmup)
    graph = bench_cuda_graph(
        model, static_input, static_output, args.iters, args.warmup
    )

    speedup = eager["per_iter_ms"] / graph["per_iter_ms"]

    print(f"Eager   per-iter: {eager['per_iter_ms']:.4f} ms  (total {eager['total_ms']:.2f} ms)")
    print(
        f"Graph   per-iter: {graph['per_iter_ms']:.4f} ms  "
        f"(p50 {graph['p50_ms']:.4f}, p99 {graph['p99_ms']:.4f} ms)"
    )
    print(f"Speedup: {speedup:.2f}x")
    print("-" * 60)
    print(
        "说明：CUDA Graph 主要减少 CPU launch 开销；"
        "当 GPU 计算本身很重时 speedup 可能较小，"
        "小 batch decode 场景通常更明显。"
    )


if __name__ == "__main__":
    main()
