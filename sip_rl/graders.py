"""Episode graders: map recorded episodes to a binary success label (+ optional score in [0, 1]).

Every grader sees the same episode records written by the env worker (sip_rl/workers.py): task text, videos of
the third-person and wrist cameras at action-chunk boundaries, and metadata. Only OracleGrader / FlipGrader read
the simulator's success flag; model graders run as a separate program in their own Python environment (loaded,
used and exited once per RL iteration, so the GPU is free again for training).
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time

import numpy as np


class Grader:
    name = "base"

    def grade(self, episodes: list[dict], work_dir: pathlib.Path) -> dict[str, dict]:
        """Return {episode_id: {"label": 0|1, "score": float|None, ...}}."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class OracleGrader(Grader):
    """LIBERO's own success check (task goal predicates satisfied at the final step)."""

    name = "oracle"

    def grade(self, episodes, work_dir):
        return {e["id"]: {"label": int(e["oracle_success_at_end"]), "score": float(e["oracle_success_at_end"])}
                for e in episodes}


class FlipGrader(Grader):
    """Control: oracle labels flipped at random with given false-positive / false-negative rates."""

    name = "flip"

    def __init__(self, fpr: float, fnr: float, seed: int = 0):
        self.fpr, self.fnr = float(fpr), float(fnr)
        self.rng = np.random.default_rng(seed)

    def grade(self, episodes, work_dir):
        out = {}
        for e in episodes:
            o = int(e["oracle_success_at_end"])
            flip = self.rng.random() < (self.fnr if o else self.fpr)
            out[e["id"]] = {"label": int(o != flip), "score": None}
        return out


class CommandGrader(Grader):
    """Runs a grader program in its own Python environment.

    One-shot mode: `<python> <script> [args] --manifest M --out O` per iteration (model loaded every time).
    Persistent mode: `<python> <script> [args] --server` is started once and keeps its model in CPU RAM; each
    request (one JSON line on stdin: {"manifest": M, "out": O}) moves the model to the GPU, grades, moves it back
    and answers with a "__SIP_DONE__" line. The GPU is therefore only used during the grading phase.
    """

    def __init__(self, name: str, python: str, script: str, args: dict | None = None, cwd: str | None = None,
                 timeout_s: float = 3600, env: dict | None = None, persistent: bool = False):
        self.name = name
        self.cmd = [python, script]
        for k, v in (args or {}).items():
            self.cmd += [f"--{k.replace('_', '-')}", str(v)]
        self.cwd, self.timeout_s, self.env = cwd, timeout_s, env or {}
        self.persistent = persistent
        self.proc = None
        self._log = None

    def _write_manifest(self, episodes, work_dir):
        work_dir.mkdir(parents=True, exist_ok=True)
        manifest, out = work_dir / f"{self.name}_manifest.jsonl", work_dir / f"{self.name}_labels.json"
        with open(manifest, "w") as f:
            for e in episodes:  # no oracle fields: the grader only gets what a real deployment would have
                f.write(json.dumps({k: e[k] for k in ("id", "task", "agentview", "wrist", "num_frames")}) + "\n")
        return manifest, out

    def _read_until(self, marker: str) -> str:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"grader {self.name} exited (code {self.proc.poll()}); see {self._log_path}")
            if line.startswith(marker):
                return line[len(marker):].strip()
            self._log.write(line)
            self._log.flush()

    def _start(self, work_dir):
        self._log_path = work_dir.parent / f"{self.name}_server.log"
        self._log = open(self._log_path, "a")  # noqa: SIM115
        self.proc = subprocess.Popen(self.cmd + ["--server"], cwd=self.cwd, env={**os.environ, **self.env},
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, text=True,
                                     bufsize=1)
        self._read_until("__SIP_READY__")

    def grade(self, episodes, work_dir):
        manifest, out = self._write_manifest(episodes, work_dir)
        t0 = time.time()
        if self.persistent:
            if self.proc is None or self.proc.poll() is not None:
                self._start(work_dir)
            self.proc.stdin.write(json.dumps({"manifest": str(manifest), "out": str(out)}) + "\n")
            self.proc.stdin.flush()
            self._read_until("__SIP_DONE__")
        else:
            log = open(work_dir / f"{self.name}.log", "w")  # noqa: SIM115
            proc = subprocess.run(self.cmd + ["--manifest", str(manifest), "--out", str(out)], cwd=self.cwd,
                                  env={**os.environ, **self.env}, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=self.timeout_s)
            log.close()
            if proc.returncode != 0:
                raise RuntimeError(f"grader {self.name} failed (exit {proc.returncode}); see {work_dir / (self.name + '.log')}")
        labels = json.loads(out.read_text())
        timing = labels.pop("__timing__", {})
        missing = [e["id"] for e in episodes if e["id"] not in labels]
        if missing:
            raise RuntimeError(f"grader {self.name} returned no label for {len(missing)} episodes, e.g. {missing[:3]}")
        for v in labels.values():
            v.setdefault("score", None)
        labels["__wall_s__"] = time.time() - t0
        labels["__timing__"] = timing
        return labels

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                self.proc.kill()


def make_grader(cfg: dict, repo: pathlib.Path) -> Grader:
    kind = cfg["name"]
    if kind == "oracle":
        return OracleGrader()
    if kind == "flip":
        return FlipGrader(cfg["fpr"], cfg["fnr"], cfg.get("seed", 0))
    cwd = None
    if cfg.get("cwd"):  # graders may write caches into their working directory; keep them out of the repo root
        cwd = repo / cfg["cwd"]
        cwd.mkdir(parents=True, exist_ok=True)
    return CommandGrader(
        name=kind,
        python=str(repo / cfg["python"]),
        script=str(repo / cfg["script"]),
        args=cfg.get("args", {}),
        cwd=str(cwd) if cwd else None,
        timeout_s=cfg.get("timeout_s", 3600),
        persistent=bool(cfg.get("persistent", False)),
    )
