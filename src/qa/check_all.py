"""Runs every check the gate needs, against a real HTTP server, then the gate itself.

    python -m qa.check_all            deterministic run (what CI does on every pull request)
    python -m qa.check_all --model    weekly real-model run (needs Ollama; see docs)

The server runs in a background thread of this process (uvicorn), on a free local port, with
traces switched on. Every suite then talks to it over HTTP, exactly as a client would:

    pytest (API tests + DeepEval)  ->  retrieval eval  ->  Ragas  ->  Promptfoo  ->  gate

A failing step doesn't stop the later ones: the gate reports everything at once. The exit code is
non-zero if any step or the gate failed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace

import httpx
import uvicorn

from belay.api import create_app
from belay.config import Settings
from qa.cases import ROOT

REPORTS = ROOT / "reports"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class BackgroundServer:
    """uvicorn in a daemon thread; ``stop()`` asks it to finish and waits."""

    def __init__(self, settings: Settings, port: int):
        config = uvicorn.Config(create_app(settings=settings), host="127.0.0.1", port=port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self.url = f"http://127.0.0.1:{port}"

    def start(self, timeout: float = 30) -> None:
        self._thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{self.url}/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.2)
        raise RuntimeError(f"server did not start within {timeout} s")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


def step(name: str, command: list[str], env: dict[str, str]) -> bool:
    print(f"\n=== {name}: {' '.join(command)}", flush=True)
    ok = subprocess.run(command, env=env, cwd=ROOT).returncode == 0
    print(f"=== {name}: {'ok' if ok else 'FAILED'}", flush=True)
    return ok


def red_team(python: str, npx: str, env: dict[str, str]) -> list[bool]:
    """Generate attacks, run them, grade the failures again, keep the fails both grades agree on."""
    tests, first, second = (
        REPORTS / "redteam-tests.yaml",
        REPORTS / "redteam-results.json",
        REPORTS / "redteam-regrade.json",
    )
    pf = [npx, "promptfoo", "redteam"]
    # Attacks are generated on this machine, not by Promptfoo's cloud service
    env = {**env, "PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION": "true", "PROMPTFOO_EVAL_TIMEOUT_MS": "300000"}
    ok = step(
        "red team: generate",
        [
            *pf,
            "generate",
            # --strict: a plugin that generates nothing fails here, instead of quietly shrinking the scan
            "--strict",
            "--max-concurrency",
            "1",
            "-c",
            "promptfooconfig.redteam.yaml",
            "-o",
            str(tests),
        ],
        env,
    )
    if not ok or not tests.exists():
        return [False]
    # Exit code 100 = some probes failed the first grade; that is what the next steps sort out
    step(
        "red team: run", [*pf, "eval", "--no-share", "--max-concurrency", "1", "-c", str(tests), "-o", str(first)], env
    )
    failed = subprocess.run(
        [python, "-m", "qa.redteam", "count", str(first)], env=env, cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    if failed not in ("", "0"):
        step(
            "red team: grade again",
            [
                *pf,
                "eval",
                "--no-share",
                "--no-cache",
                "--max-concurrency",
                "1",
                "-c",
                str(tests),
                "--filter-failing",
                str(first),
                "-o",
                str(second),
            ],
            env,
        )
    return [step("red team: confirm", [python, "-m", "qa.redteam", "confirm", str(first), str(second)], env)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="store_true", help="answer with Ollama and run the LLM-judged checks")
    parser.add_argument("--judge-limit", type=int, default=12, help="judged cases (model run only)")
    args = parser.parse_args(argv)

    REPORTS.mkdir(exist_ok=True)
    for old in REPORTS.glob("*"):
        if old.is_file():
            old.unlink()  # reports of an earlier run must never reach this gate

    settings = replace(Settings.from_env(), trace_path=REPORTS / "traces.jsonl")
    if args.model:
        settings = replace(settings, generator="ollama", embedder="ollama")
    server = BackgroundServer(settings, free_port())
    server.start()
    env = {
        **os.environ,
        "BELAY_BASE_URL": server.url,
        "BELAY_GENERATOR": settings.generator,
        "PYTHONPATH": str(ROOT / "src"),
        # No usage data leaves the run
        "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
        "RAGAS_DO_NOT_TRACK": "true",
        "PROMPTFOO_DISABLE_TELEMETRY": "1",
        "PROMPTFOO_DISABLE_UPDATE": "1",
        "JUDGE_LIMIT": str(args.judge_limit),
    }
    python = sys.executable
    npx = shutil.which("npx") or "npx"
    results = []
    try:
        pytest_args = [python, "-m", "pytest", "--junitxml", str(REPORTS / "junit.xml")]
        if args.model:
            # Only the eval suites: the API tests are about the HTTP layer, which doesn't change
            # with the generator, and 60 rate-limit calls through a CPU model would take minutes.
            # A model's single answers vary, so a failed case here is information, not the
            # verdict: the model gate decides on the averages it recorded.
            step("pytest", [*pytest_args, "--model", "tests/evals"], env)
        else:
            results.append(step("pytest", pytest_args, env))
        retrieval = [python, "-m", "qa.retrieval_eval"] + (["--embedder", "ollama"] if args.model else [])
        results.append(step("retrieval", retrieval, env))
        ragas = [python, "-X", "utf8", "-m", "qa.ragas_eval"] + (
            ["--model", "--limit", str(args.judge_limit)] if args.model else []
        )
        results.append(step("ragas", ragas, env))
        if args.model:
            results.append(step("stability", [python, "-m", "qa.stability"], env))
            results.extend(red_team(python, npx, env))
        else:
            promptfoo = [
                npx,
                "promptfoo",
                "eval",
                "--no-share",
                "-c",
                "promptfooconfig.yaml",
                "-o",
                str(REPORTS / "promptfoo.json"),
            ]
            results.append(step("promptfoo", promptfoo, env))
    finally:
        server.stop()

    gate_config = ROOT / "configs" / ("model_gate.toml" if args.model else "quality_gate.toml")
    results.append(step("gate", [python, "-m", "qa.gate", "--config", str(gate_config)], env))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
