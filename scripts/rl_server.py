#!/usr/bin/env python3
"""
rl_server.py — Unified RL server entry point for all models.

Usage:
  python3 scripts/rl_server.py --model-id m0_0 --mode train --train-steps 50000
  python3 scripts/rl_server.py --model-id m2   --mode eval  --eval-steps 20000
"""

import argparse, importlib, os, struct, sys, time

# Allow `from models.<id> import ...` when invoked from any directory.
sys.path.insert(0, os.path.dirname(__file__))

from models.common import (
    ACTION_COLUMNS, ACTION_SIZE, DEFAULT_TRAIN_STEPS,
    PlateauDetector, DQNAgent, compute_reward,
    create_shm, shm_write_action,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id",    required=True,
                    help="Model ID: m0_0, m1_0, m1_1, m2")
    ap.add_argument("--mode",        choices=["train", "eval"], default="train")
    ap.add_argument("--model",       default=None,
                    help="Checkpoint path (default: derived from model-id)")
    ap.add_argument("--eval-steps",  type=int, default=20_000)
    ap.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    ap.add_argument("--results-dir", default=".")
    ap.add_argument("--no-plateau",  action="store_true",
                    help="Disable coverage-plateau early stopping")
    args = ap.parse_args()

    mod = importlib.import_module(f"models.{args.model_id}")

    env_key = f"RL_{args.model_id.upper()}_MODEL_PATH"
    if args.model is None:
        args.model = os.environ.get(env_key, mod.MODEL_PATH_DEFAULT)

    is_eval = args.mode == "eval"
    tag     = "EVAL" if is_eval else "TRAIN"
    label   = mod.LABEL
    os.makedirs(args.results_dir, exist_ok=True)
    mpath = os.path.join(args.results_dir,
                         f"rl_metrics_{args.model_id}_{'eval' if is_eval else 'train'}.csv")

    print(f"[+] {label}  mode={tag}  state={mod.STATE_SIZE}  actions={ACTION_SIZE}  "
          f"train_steps={args.train_steps}"
          + ("  [no-plateau]" if args.no_plateau else ""))

    shm = create_shm(mod.SHM_PATH, mod.SHM_SIZE)
    agent = DQNAgent(mod.STATE_SIZE, mod.HIDDEN_LAYERS, label,
                     eval_mode=is_eval,
                     decay_steps=int(args.train_steps * 0.6))
    agent.load(args.model)
    plateau = (PlateauDetector(grace_steps=int(args.train_steps * 0.7))
               if (not is_eval and not args.no_plateau) else None)
    shm_write_action(shm, 46, 1, mod.ACTION_OFF, mod.ACTION_SEQ_OFF)

    zero_d = mod.zero_state_data()
    csv_header = ("step,reward,coverage_term,crash_term,loss,epsilon,"
                  f"coverage,crashes,action{mod.CSV_EXTRA_HEADER},elapsed_seconds\n")
    with open(mpath, "w") as f:
        f.write(csv_header)

    print(f"[+] {label} ready.  metrics={mpath}")

    pcov = 0; pcr = 0; pact = 46; pstate = mod.build_state(zero_d, args.train_steps)
    last_seq = 0; aseq = 1; step = 0; cov_start = None
    stop = "user interrupt"; cov = 0; cr = 0
    d = zero_d
    start_time = time.time()

    try:
        while True:
            if is_eval and step >= args.eval_steps:      stop = "eval limit";  break
            if not is_eval and step >= args.train_steps:  stop = "train limit"; break

            while True:
                shm.seek(mod.STATE_SEQ_OFF)
                cur = struct.unpack("=I", shm.read(4))[0]
                if cur != last_seq: break
                time.sleep(0.0001)

            last_seq = cur; d = mod.shm_read(shm, mod.SHM_SIZE)
            cov = d["coverage"]; ne = d["new_edges"]; cr = d["crashes"]
            if is_eval and cov_start is None: cov_start = cov

            state = mod.build_state(d, args.train_steps)
            rew, comps = compute_reward(cov, pcov, cr, pcr)
            loss = 0.0
            if step > 0:
                agent.remember(pstate, pact, rew, state); loss = agent.train_step()

            act = agent.select_action(state); aseq += 1
            shm_write_action(shm, act, aseq, mod.ACTION_OFF, mod.ACTION_SEQ_OFF)

            if step % 100 == 0:
                extra_csv = mod.csv_extra_fields(d, args)
                elapsed = time.time() - start_time
                with open(mpath, "a") as f:
                    f.write(f"{step},{rew:.4f},{comps['coverage_term']:.4f},"
                            f"{comps['crash_term']:.4f},{loss:.6f},{agent.epsilon:.4f},"
                            f"{cov},{cr},{act}{extra_csv},{elapsed:.2f}\n")
                extra_log = mod.log_extra(d, args)
                print(f"[{label}-{tag}:{step:>7}] cov={cov:<5} ne={ne:<3} cr={cr} "
                      f"act={act:<2}({ACTION_COLUMNS[act]:<30}) "
                      f"\u03b5={agent.epsilon:.3f} r={rew:+.2f} loss={loss:.5f}"
                      + (f" {extra_log}" if extra_log else ""))

            if not is_eval and step > 0 and step % 1000 == 0:
                agent.save(args.model)
            if not is_eval and plateau and plateau.update(cov, agent.epsilon, step):
                stop = "coverage plateau"; break

            pstate = state; pact = act; pcov = cov; pcr = cr; step += 1

    except KeyboardInterrupt:
        stop = "user interrupt"
    finally:
        if not is_eval: agent.save(args.model)
        print(f"\n{'='*58}\n  {label} {tag} done \u2014 {stop}")
        print(f"  steps={step:,}  cov={cov}  crashes={cr}  \u03b5={agent.epsilon:.4f}")
        mod.exit_summary(d, step, cov, cr, agent.epsilon, tag)
        print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
