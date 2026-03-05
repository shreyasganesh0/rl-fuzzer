#!/usr/bin/env python3
"""
benchmark_latency.py — Measure per-component latency in the RL server loop.

Profiles:
  1. SHM read (struct unpack)
  2. State construction (numpy array build + normalization)
  3. Reward computation
  4. Replay buffer push + epsilon decay
  5. DQN forward pass (action selection) — EVAL mode (just inference)
  6. DQN training step (sample + forward + backward + optimizer step)
  7. SHM write (action write back)
  8. Full loop iteration (all of the above combined)

Usage:
  python3 scripts/benchmark_latency.py [--model-id m0_0] [--iters 10000] [--device cpu]
"""

import argparse, importlib, os, struct, sys, time, random, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from models.common import (
    ACTION_SIZE, BATCH_SIZE, DQNAgent, compute_reward,
)


def bench(fn, n, warmup=100):
    """Run fn() n+warmup times, return (mean_us, std_us, min_us, max_us) for the last n."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) / 1000.0)  # ns -> us
    arr = np.array(times)
    return arr.mean(), arr.std(), arr.min(), arr.max(), np.median(arr), np.percentile(arr, 99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="m0_0")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    mod = importlib.import_module(f"models.{args.model_id}")
    label = mod.LABEL
    state_size = mod.STATE_SIZE
    hidden = mod.HIDDEN_LAYERS

    print(f"Benchmarking {label}: state_size={state_size}, hidden={hidden}, "
          f"actions={ACTION_SIZE}, iters={args.iters}, device={args.device}")
    print(f"{'='*80}")

    # Force device
    import torch
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "" if args.device == "cpu" else "0")

    # Create agent in eval mode (no training overhead for inference bench)
    agent_eval = DQNAgent(state_size, hidden, label, eval_mode=True)
    agent_train = DQNAgent(state_size, hidden, label, eval_mode=False, decay_steps=50000)

    # Pre-fill replay buffer so training step actually runs
    for _ in range(BATCH_SIZE + 50):
        s = np.random.randn(state_size).astype(np.float32)
        ns = np.random.randn(state_size).astype(np.float32)
        agent_train.remember(s, random.randrange(ACTION_SIZE),
                             random.uniform(-1, 1), ns)

    # Fake SHM data
    fake_shm_data = b'\x00' * mod.SHM_SIZE

    # Fake state data dict
    zero_d = mod.zero_state_data()
    # Make a "realistic" data dict
    real_d = dict(zero_d)
    real_d["coverage"] = 1234
    real_d["new_edges"] = 5
    real_d["crashes"] = 2

    state = mod.build_state(real_d, 50000)

    results = {}

    # 1. SHM read (struct unpack)
    def bench_shm_read():
        struct.unpack_from("=I", fake_shm_data, 0)
        struct.unpack_from("=I", fake_shm_data, 4)
        struct.unpack_from("=I", fake_shm_data, 8)
        struct.unpack_from("=I", fake_shm_data, 12)
        if hasattr(mod, 'TOTAL_EXECS_OFF'):
            struct.unpack_from("=Q", fake_shm_data, mod.TOTAL_EXECS_OFF)
        if state_size > 10:  # M2 has extra arrays
            struct.unpack_from(f"={ACTION_SIZE}f", fake_shm_data, 32)
            struct.unpack_from(f"={ACTION_SIZE}f", fake_shm_data, 220)

    results["SHM read (struct unpack)"] = bench(bench_shm_read, args.iters)

    # 2. State construction
    def bench_build_state():
        mod.build_state(real_d, 50000)

    results["State build (numpy)"] = bench(bench_build_state, args.iters)

    # 3. Reward computation
    def bench_reward():
        compute_reward(1234, 1230, 2, 1)

    results["Reward computation"] = bench(bench_reward, args.iters)

    # 4. Replay buffer push + decay
    s_sample = np.random.randn(state_size).astype(np.float32)
    ns_sample = np.random.randn(state_size).astype(np.float32)
    def bench_remember():
        agent_train.remember(s_sample, 5, 0.5, ns_sample)

    results["Replay push + decay"] = bench(bench_remember, args.iters)

    # 5. DQN forward pass (inference only — eval mode)
    def bench_forward_eval():
        agent_eval.select_action(state)

    results["DQN forward (eval)"] = bench(bench_forward_eval, args.iters)

    # 6. DQN forward pass (train mode — includes epsilon check)
    agent_train.epsilon = 0.0  # force network inference path
    def bench_forward_train():
        agent_train.select_action(state)

    results["DQN forward (train, eps=0)"] = bench(bench_forward_train, args.iters)

    # 7. DQN training step (sample + forward + backward + optim)
    def bench_train_step():
        agent_train.train_step()

    results["DQN train_step (batch)"] = bench(bench_train_step, args.iters)

    # 8. SHM write (action write back)
    import mmap as mmap_mod
    shm_path = "/tmp/rl_bench_shm"
    with open(shm_path, "w+b") as fd:
        fd.write(b'\x00' * 128)
        fd.flush()
    fd = open(shm_path, "r+b")
    shm = mmap_mod.mmap(fd.fileno(), 128)
    seq = [1]
    def bench_shm_write():
        shm.seek(68); shm.write(struct.pack("=i", 5))
        shm.seek(64); shm.write(struct.pack("=I", seq[0])); shm.flush()
        seq[0] += 1

    results["SHM write (action)"] = bench(bench_shm_write, args.iters)
    shm.close(); fd.close(); os.unlink(shm_path)

    # 9. Full loop (simulated — everything except actual SHM wait)
    pcov, pcr, pact = 1230, 1, 5
    pstate = state.copy()
    def bench_full_loop():
        # shm_read
        bench_shm_read()
        # build_state
        s = mod.build_state(real_d, 50000)
        # reward
        rew, _ = compute_reward(1234, pcov, 2, pcr)
        # remember + train
        agent_train.remember(pstate, pact, rew, s)
        agent_train.train_step()
        # select action
        agent_train.select_action(s)
        # shm_write (inline)
        struct.pack("=i", 5)
        struct.pack("=I", 1)

    results["Full loop (no SHM wait)"] = bench(bench_full_loop, args.iters)

    # Print results
    print(f"\n{'Component':<35} {'Mean':>10} {'Std':>10} {'Min':>10} "
          f"{'Median':>10} {'P99':>10} {'Max':>10}  (all in microseconds)")
    print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for name, (mean, std, mn, mx, med, p99) in results.items():
        print(f"{name:<35} {mean:>10.1f} {std:>10.1f} {mn:>10.1f} "
              f"{med:>10.1f} {p99:>10.1f} {mx:>10.1f}")

    # Summary analysis
    fwd = results["DQN forward (eval)"][0]
    train = results["DQN train_step (batch)"][0]
    full = results["Full loop (no SHM wait)"][0]
    overhead = full - fwd  # non-inference overhead

    print(f"\n{'='*80}")
    print(f"ANALYSIS")
    print(f"{'='*80}")
    print(f"DQN inference latency:       {fwd:>10.1f} us  ({fwd/1000:.3f} ms)")
    print(f"DQN training step latency:   {train:>10.1f} us  ({train/1000:.3f} ms)")
    print(f"Full loop latency:           {full:>10.1f} us  ({full/1000:.3f} ms)")
    print(f"Non-DQN overhead:            {overhead:>10.1f} us  ({overhead/1000:.3f} ms)")
    print(f"")
    print(f"Network architecture: {state_size} -> {' -> '.join(map(str,hidden))} -> {ACTION_SIZE}")
    total_params = sum(p.numel() for p in agent_eval.online.parameters())
    print(f"Total parameters:    {total_params:,}")
    print(f"")

    # Estimate C native inference
    # Pure C forward pass: matrix multiply + ReLU, no Python/torch overhead
    # Rough estimate: ~1 FLOP per MAC, 2*params FLOPs total
    flops = 2 * total_params
    # Modern CPU: ~10 GFLOPS single-thread conservative
    est_c_forward_us = flops / 10e9 * 1e6  # very rough
    print(f"Estimated C native forward pass:")
    print(f"  FLOPs per inference:  {flops:,}")
    print(f"  @ 10 GFLOPS (1 core): {est_c_forward_us:.2f} us")
    print(f"  Speedup vs Python:    ~{fwd/max(est_c_forward_us,0.01):.0f}x")
    print(f"")

    # Per-step time budget
    print(f"Context: AFL++ exec speed is typically 100-10000 execs/sec")
    print(f"  At 1000 execs/sec: 1000 us budget per exec")
    print(f"  Python RL overhead: {full:.0f} us ({full/1000*100:.1f}% of budget @ 1000 exec/s)")
    print(f"  Estimated C overhead: ~{est_c_forward_us:.1f} us ({est_c_forward_us/1000*100:.2f}% of budget)")


if __name__ == "__main__":
    main()
