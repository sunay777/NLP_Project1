"""Run the v2 experiment queue (experiments/jobs_v2.py) with N parallel workers.

Each job is one run_collapse trajectory on CPU with ONE torch thread: many
small single-thread processes give ~6x the throughput of one multi-threaded
process for a model this small (measured on an M4: 8 x 1-thread = one extended
generation every ~35 s vs ~170 s for a single process).

Restart-safe: a job is done when its --out JSON holds the finished trajectory;
re-running the queue skips it. Ctrl-C stops the workers (finished trajectories
are kept; the interrupted ones re-run from the start next time). The Mac is
kept awake with caffeinate while the queue runs.

  python -m experiments.run_queue --benchmark       # time this machine, print an ETA
  python -m experiments.run_queue                   # run everything (stages 1 -> 3)
  python -m experiments.run_queue --stages 1        # only the core stage
  python -m experiments.run_queue --status          # progress + ETA (from another terminal)
  python -m experiments.run_queue --status --only pool zipf   # progress of one Mac's share
  python -m experiments.run_queue --workers 10      # override the worker count
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

from experiments.jobs_v2 import jobs, ROOT

QDIR = f"{ROOT}/_queue"
STATUS = f"{QDIR}/status.json"


def is_done(j):
    if not os.path.exists(j["out"]):
        return False
    try:
        d = json.load(open(j["out"]))
    except (json.JSONDecodeError, OSError):
        return False
    return any(r["real_fraction"] == j["real_fraction"] and r["seed"] == j["seed"]
               and len(r["records"]) == j["generations"] for r in d.get("runs", []))


def default_workers():
    try:
        n = int(subprocess.check_output(["sysctl", "-n", "hw.ncpu"]).strip())
    except Exception:
        n = os.cpu_count() or 4
    return max(1, n - 1)


def write_status(state):
    os.makedirs(QDIR, exist_ok=True)
    tmp = STATUS + ".tmp"
    json.dump(state, open(tmp, "w"), indent=1)
    os.replace(tmp, STATUS)


def fmt_h(s):
    return f"{s / 3600:.1f} h"


def status(only=None):
    J = [j for j in jobs() if not only or any(o in j["name"] for o in only)]
    done = [j for j in J if is_done(j)]
    st = json.load(open(STATUS)) if os.path.exists(STATUS) else {}
    rate = st.get("sec_per_unit")
    left = sum(j["est_units"] for j in J if not is_done(j))
    print(f"done {len(done)}/{len(J)} jobs; remaining {left:.0f} units"
          + (f"; ETA ~{fmt_h(left * rate)} at {rate:.0f} s/unit" if rate else ""))
    for stg in sorted({j["stage"] for j in J}):
        js = [j for j in J if j["stage"] == stg]
        print(f"  stage {stg}: {sum(is_done(j) for j in js)}/{len(js)} jobs")
    for r in st.get("running", []):
        print(f"  running: {r['name']} rf={r['real_fraction']} s={r['seed']} "
              f"since {time.strftime('%H:%M', time.localtime(r['started']))}")
    for f in st.get("failed", []):
        print(f"  FAILED: {f['out']} (see {f['log']})")


def benchmark(workers, steps=300):
    """Time `workers` concurrent 1-thread extended trainings; returns s per unit."""
    code = ("import sys,time,torch;sys.path.insert(0,'.');torch.set_num_threads(1)\n"
            "from src.config import Config\nfrom src import data as D\nfrom src.train import train_model\n"
            f"c=Config(mode='extended',steps={steps},eval_every=10**9,device='cpu')\n"
            "t=time.time();train_model(c,lambda b,r,d:D.make_batch(c,b,r,d),log=False)\n"
            "print(time.time()-t)")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    ps = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=env)
          for _ in range(workers)]
    secs = [float(p.communicate()[0].strip().splitlines()[-1]) for p in ps]
    per_gen = sum(secs) / len(secs) * 6000 / steps * 1.12      # + eval/generation/gate overhead
    return per_gen / workers, per_gen


def run(args):
    J = [j for j in jobs() if j["stage"] in args.stages]
    if args.only:
        J = [j for j in J if any(o in j["name"] for o in args.only)]
    todo = [j for j in J if not is_done(j)]
    # within a stage: longest first (fewer stragglers at the end)
    todo.sort(key=lambda j: (j["stage"], -j["est_units"]))
    print(f"{len(J) - len(todo)}/{len(J)} jobs already done; {len(todo)} to run "
          f"({sum(j['est_units'] for j in todo):.0f} units) on {args.workers} workers")
    if not todo:
        return
    awake = subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())]) \
        if sys.platform == "darwin" else None
    running, failed, finished = {}, [], []
    t_start, units_done = time.time(), 0.0
    stop = {"flag": False}

    def on_sigint(*_):
        stop["flag"] = True
        print("\nstopping: terminating running jobs (finished trajectories are kept)")
        for p, _, _ in running.values():
            p.terminate()
    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    def launch(j):
        os.makedirs(os.path.dirname(j["out"]), exist_ok=True)
        log = os.path.splitext(j["out"])[0] + ".log"
        cmd = [sys.executable, "-m", "experiments.run_collapse", *j["args"],
               "--device", "cpu", "--threads", "1", "--out", j["out"]]
        f = open(log, "a")
        f.write(f"\n# {time.ctime()}  {' '.join(cmd)}\n")
        f.flush()
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "VECLIB_MAXIMUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        running[p.pid] = (p, j, time.time())

    queue = list(todo)
    while (queue or running) and not stop["flag"]:
        while queue and len(running) < args.workers and not stop["flag"]:
            launch(queue.pop(0))
        time.sleep(5)
        for pid in list(running):
            p, j, t0 = running[pid]
            if p.poll() is None:
                continue
            del running[pid]
            if p.returncode == 0 and is_done(j):
                finished.append(j)
                units_done += j["est_units"]
                print(f"[{time.strftime('%H:%M')}] done {j['name']} rf={j['real_fraction']} "
                      f"s={j['seed']} in {(time.time() - t0) / 60:.0f} min", flush=True)
            elif not stop["flag"]:
                j.setdefault("_tries", 0)
                j["_tries"] += 1
                log = os.path.splitext(j["out"])[0] + ".log"
                if j["_tries"] <= 1:
                    print(f"job failed, will retry once: {j['out']} (log {log})", flush=True)
                    queue.append(j)
                else:
                    failed.append({"out": j["out"], "log": log})
                    print(f"job FAILED twice: {j['out']} (log {log})", flush=True)
        el = time.time() - t_start
        rate = el / units_done if units_done else None          # wall s per unit
        left = sum(j["est_units"] for j in queue) + sum(j["est_units"] for _, j, _ in running.values())
        write_status({"updated": time.time(), "workers": args.workers,
                      "sec_per_unit": rate, "remaining_units": left,
                      "running": [{"name": j["name"], "real_fraction": j["real_fraction"],
                                   "seed": j["seed"], "started": t0}
                                  for _, j, t0 in running.values()],
                      "finished_this_session": len(finished), "failed": failed})
    if awake:
        awake.terminate()
    print(f"queue finished: {len(finished)} jobs this session, {len(failed)} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--only", nargs="*", default=None, help="substring filter on job names")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    if a.status:
        return status(a.only)
    if a.benchmark:
        print(f"benchmarking {a.workers} parallel workers (about 1-2 minutes)...")
        spu, per_gen = benchmark(a.workers)
        J = [j for j in jobs() if not is_done(j)]
        for stg in sorted({j["stage"] for j in J}):
            u = sum(j["est_units"] for j in J if j["stage"] == stg)
            print(f"  stage {stg}: {u:.0f} units -> ~{fmt_h(u * spu)}")
        tot = sum(j["est_units"] for j in J)
        print(f"one generation takes ~{per_gen:.0f} s per worker; with {a.workers} workers "
              f"~{spu:.0f} s per unit overall\nTOTAL remaining: {tot:.0f} units -> ~{fmt_h(tot * spu)}"
              " (+ retries; the first stage alone is the core result)")
        return
    if a.dry_run:
        for j in jobs():
            if j["stage"] in a.stages:
                print(("DONE " if is_done(j) else "todo ") + j["out"], " ".join(j["args"]))
        return
    run(a)


if __name__ == "__main__":
    main()
