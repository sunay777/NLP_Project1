# Running the v2 queue on two Macs (3/5 Matthew, 2/5 Sunay)

The v2 queue (`experiments/jobs_v2.py`, 281 jobs, 3107 units; 1 unit = one 6000-step
extended generation) is split by experiment name between two Macs. Every job is an
independent training run that writes its own `results/v2/<experiment>/rf<λ>_s<seed>.*`
files, so the two halves never touch the same file. Setup is the same as in
[RUN_V2.md](RUN_V2.md); only the `start` step differs.

Deadline: **27 Oct 2026, 17:00**.

## Who runs what

| | Matthew's Mac | Sunay's Mac |
|---|---|---|
| Share | **1841 units, 180 jobs (59%)** | **1266 units, 101 jobs (41%)** |
| Stage 1 (headline) | all of it: `main_base`, `main_ext` (seeds 0-7), `temp_T0.7/1.3/1.6`, `pulse_ext`, `pulse_base` | none |
| Stage 2 | `scrub_offctx`, `scrub_labels`, `accumulate`, `fulllm_synthctx` | `fulllm_promptreal`, `pool5k`, `pool80k`, `zipf1`, `zipf2` |
| Stage 3 | `main_ext` (seeds 8-9) | `arch_L1/L2d128/L2mlp/L3/L4`, `task_K6`, `task_S32` |
| `--only` filter | `main_ pulse temp scrub accumulate fulllm_synthctx` | `fulllm_promptreal pool zipf arch task` |

Checked with `experiments/jobs_v2.py`: the two filters overlap on 0 jobs and cover all 281.
Matthew's Mac takes all of stage 1, so the headline results do not depend on the second Mac.

## What is running on Matthew's Mac

- A temperature-calibration rerun (seed 100) duplicated Sunay's committed calibration.
  7 of its 8 runs finished; the last (T=1.6, λ=0, runaway regime) was stopped once
  Sunay's calibration was pushed. They are kept as a reproducibility check in
  `results/v2/temperature_calib_rerun_matthew/` (local only, not committed).
  The pre-registration used is Sunay's committed `paper/PREDICTION.json` / `.md`.
- The queue, started with Matthew's filter below. Measured speed on this Mac is about
  6 minutes per generation per core, so its share takes roughly **18-21 hours**
  (stage 1 runs first).

## Commands

Run everything from the `NLP_Project1` folder, on the charger with the lid open.
Do `git pull` first so `paper/PREDICTION.json` is present: the temperature jobs read it.

### Both Macs, once

```bash
git pull
```

```bash
./run_v2.sh setup
```

`setup` must end with `23/23 job types ran`. It needs Python 3.10-3.13; if only a newer
Python is installed, `uv python install 3.12` provides one without admin rights.

Make sure no queue is already running (this should print nothing):

```bash
pgrep -fl experiments.run_queue
```

### Start: Matthew's Mac

```bash
mkdir -p results/v2/_queue && nohup .venv/bin/python -m experiments.run_queue --only main_ pulse temp scrub accumulate fulllm_synthctx >> results/v2/_queue/queue.log 2>&1 &
```

### Start: Sunay's Mac

```bash
mkdir -p results/v2/_queue && nohup .venv/bin/python -m experiments.run_queue --only fulllm_promptreal pool zipf arch task >> results/v2/_queue/queue.log 2>&1 &
```

Do not use plain `./run_v2.sh start` on either Mac: it runs all 281 jobs and would
duplicate the other Mac's share.

### While it runs (both Macs)

- Progress and ETA: `./run_v2.sh status`
- Pause: `./run_v2.sh stop`. To resume, rerun the same **Start** command for that Mac.
  Finished jobs are kept; only the jobs in progress restart.
- A job that fails is retried once automatically; `status` lists anything marked `FAILED`
  with its log file.

## Bringing the results together

The analysis runs on Sunay's Mac, so Matthew's results go to Sunay.

1. When Matthew's stage-1 jobs finish, send the numbers (a few MB) for a first look:

   ```bash
   ./run_v2.sh packjson
   ```

2. When Matthew's whole share finishes, send everything, including the model weights and
   saved pools that `lineage`, `attribution` and `mechanism` need (several hundred MB):

   ```bash
   ./run_v2.sh pack
   ```

Both write a `.tar.gz` to the Desktop. Send it by AirDrop or Google Drive, not GitHub.
Sunay unpacks it from the repo root, skipping Matthew's queue log and status file, and
never replacing a file that is newer on his Mac (for example an analysis he has rerun):

```bash
tar -xzf <file> --exclude 'results/v2/_queue' --keep-newer-files
```

Matthew's own jobs are new files on Sunay's Mac, so they all extract.

The analysis commands are in `README.txt` under "Analysis of the v2 runs".
