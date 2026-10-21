"""Two-minute check that every v2 job type runs on this machine.

Runs one tiny trajectory (60 steps, 2 generations) per distinct experiment in
experiments/jobs_v2.py into a temporary folder and checks the output JSON. It
tests the plumbing only: 60 steps is far too few to learn the task, so the
printed metrics are meaningless.

Run:  python -m experiments.smoke_test
"""
import json
import os
import subprocess
import sys
import tempfile
import time

from experiments.jobs_v2 import jobs


def _shrink(args):
    out, skip = [], False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a in ("--steps", "--generations", "--max_retries", "--pool_size"):
            out += [a, {"--steps": "60", "--generations": "2", "--max_retries": "0",
                        "--pool_size": "2000"}[a]]
            skip = True
            continue
        if a == "--pulse_gen":           # keep the pulse inside the 2 generations
            out += [a, "1"]
            skip = True
            continue
        out.append(a)
    if "--pool_size" not in out:
        out += ["--pool_size", "2000"]
    return out


def main():
    seen, cases = set(), []
    for j in jobs():
        if j["name"] not in seen:
            seen.add(j["name"])
            cases.append(j)
    tmp = tempfile.mkdtemp(prefix="v2smoke_")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    procs = []
    t0 = time.time()
    for j in cases:
        out = os.path.join(tmp, j["name"] + ".json")
        cmd = [sys.executable, "-m", "experiments.run_collapse", *_shrink(j["args"]),
               "--device", "cpu", "--threads", "1", "--out", out]
        log = open(out.replace(".json", ".log"), "w")
        procs.append((j["name"], out, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)))
        while sum(p.poll() is None for *_, p in procs) >= max(1, (os.cpu_count() or 4) - 1):
            time.sleep(0.5)
    bad = 0
    for name, out, p in procs:
        p.wait()
        ok = p.returncode == 0 and os.path.exists(out) and \
            len(json.load(open(out))["runs"][0]["records"]) >= 1
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f"  (log: {out[:-5]}.log)"))
    print(f"{len(procs) - bad}/{len(procs)} job types ran in {time.time() - t0:.0f} s"
          + ("" if not bad else " -- send the FAIL logs back before running the queue"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
