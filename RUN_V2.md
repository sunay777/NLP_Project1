# Running the v2 experiments on another Mac

These steps run every v2 experiment (about 280 training runs) on one Mac and send the results back. You don't need to understand the project. Each step is one command in Terminal.

**Needs:** an Apple-silicon Mac (M1 or newer), Python 3.12 or 3.13 (`python3.12 --version` to check; numpy 2.5 needs ≥ 3.12), about 2 GB of free disk space, and the charger.

## 1. Get the code and set it up (about 5 minutes)

```bash
cd ~/Desktop
git clone https://github.com/sunay777/NLP_Project1.git
cd NLP_Project1
./run_v2.sh setup
```

`setup` creates a private Python environment in `.venv` (it doesn't touch the system Python) and installs pinned versions of torch, numpy and matplotlib. Then it runs a **2-minute smoke test** that tries every experiment type at a tiny size. It should end with `23/23 job types ran`. If any line says `FAIL`, stop and send that log file back.

No `python3`? Install it from https://www.python.org/downloads/macos/ (3.12), then rerun `./run_v2.sh setup`.

## 2. Check how long it will take (2 minutes)

```bash
./run_v2.sh bench
```

This prints the expected time per stage on this Mac:

| Stage | What | Rough size |
|---|---|---|
| 1 | Core results: main sweep, temperature test, poison pulse | ~45% of the total |
| 2 | Mechanism and data experiments | ~28% |
| 3 | Model sizes, other tasks, extra seeds | ~28% |

On a 10-core M4 the whole queue is roughly 30–40 hours. Stage 1 alone is roughly 13–17 hours.

## 3. Start it

```bash
./run_v2.sh start
```

The queue runs in the background, one training run per CPU core. You can close the Terminal window. While it runs:

- **Keep the Mac on the charger with the lid open.** The queue stops idle sleep itself, but closing the lid on battery still sleeps the Mac.
- The fans will be loud and the Mac will feel slower. Light use like browsing is fine, but it slows the queue.
- To pause, run `./run_v2.sh stop`. To continue later, run `./run_v2.sh start` again. Finished runs are kept, and only the runs that were in progress restart from scratch.
- A restart or crash is safe: rerun `./run_v2.sh start`.

To check progress at any time:

```bash
./run_v2.sh status
```

If you need the Mac back sooner, `./run_v2.sh start 1` runs only stage 1, the part that matters most.

## 4. Send the results back

**Quick numbers** (a few MB). Send this when stage 1 finishes, and again at the end:

```bash
./run_v2.sh packjson
```

**Everything**, including the trained model weights (several hundred MB), once the queue finishes or is stopped:

```bash
./run_v2.sh pack
```

Both commands put a `.tar.gz` file on the Desktop. Send it by AirDrop or Google Drive. Please don't push to GitHub.

## If something goes wrong

- `./run_v2.sh status` lists any `FAILED` job with its log file. A failed job is retried once automatically.
- **"queue already running"** means it's still going. Use `status` to check on it.
- Everything the queue writes lives in `results/v2/`. Deleting that folder resets the whole queue.

---

## For the project owner: what the queue runs

The jobs are defined in `experiments/jobs_v2.py` (`python -m experiments.jobs_v2` prints the table). Each job is one `experiments.run_collapse` trajectory: one condition, one real fraction, one seed. It runs on CPU with 1 thread and writes `results/v2/<experiment>/rf<λ>_s<seed>.json`, a `.log`, and one `.pt` per generation.

All jobs share these settings:
- `--gen0_pool`, which removes the finite-data confound.
- The strict gate: `--gate 0.98 --gate_causal 0.5`.
- `--gen_queries all` and `--save_models all`.

The temperature jobs read `paper/PREDICTION.json` (the pre-registration) to add λ* ± 0.1 grid points whenever a(T) > 1. **That file must be committed before the friend clones the repo.**
