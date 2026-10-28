"""scripts/calibrate_qwen38.py's recommendation rule and env file, and how serve-qwen38.sh picks
DECODE_FETCH / CPU_DECODE_THREADS: explicit env > the calibration file for this GPU set > defaults.
CPU only: the server is a stub that records the environment it was started with."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _calibrate_module():
    spec = importlib.util.spec_from_file_location("calibrate_qwen38", SCRIPTS / "calibrate_qwen38.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cal = _calibrate_module()


def _table(rows):
    return {p: dict(zip(("short", "long"), v)) for p, v in rows.items()}


def test_pick_prefers_the_higher_share_within_noise():
    t = _table({1.0: (35, 25), 0.5: (46, 36), 0.25: (53, 43), 0.15: (55.8, 45.6), 0.1: (56.3, 46), 0.05: (54.8, 44)})
    p, why = cal.choose_fetch(t, ["short", "long"])
    assert p == 0.15 and "0.1" in why


def test_pick_is_off_when_cpu_help_does_not_pay():
    # two GPUs: the best share loses, or wins too little at one depth
    assert cal.choose_fetch(_table({1.0: (92, 62), 0.5: (89, 61), 0.1: (84, 58)}), ["short", "long"])[0] == 1.0
    assert cal.choose_fetch(_table({1.0: (50, 40), 0.5: (60, 41)}), ["short", "long"])[0] == 1.0


def test_pick_rejects_a_share_that_loses_at_one_depth():
    t = _table({1.0: (40, 30), 0.1: (70, 29), 0.5: (52, 34)})
    assert cal.choose_fetch(t, ["short", "long"])[0] == 0.5


def test_threads_fewest_within_noise():
    t = {7: {"short": 56.0}, 8: {"short": 56.4}, 3: {"short": 50.0}}
    assert cal.choose_threads(t, ["short"]) == 7


def _snap(tmp_path, fmt="nvfp4"):
    snap = tmp_path / "snap"
    snap.mkdir(exist_ok=True)
    (snap / "freetoken_weight.json").write_text(json.dumps({"quant_format": fmt}))
    return snap


def _serve(tmp_path, gpus, env_extra=None, fmt="nvfp4"):
    """Run serve-qwen38.sh against a stub ``ft``; return (its FREETOKEN_* env, stderr)."""
    out = tmp_path / "ft-env.txt"
    stub = tmp_path / "ft"
    stub.write_text(f"#!/bin/bash\nenv | grep '^FREETOKEN_\\(MOE_DECODE_FETCH\\|CPU_DECODE_THREADS\\)=' > {out}\n")
    stub.chmod(0o755)
    env = {k: v for k, v in os.environ.items()
           if k not in ("DECODE_FETCH", "CPU_DECODE_THREADS", "CALIBRATION", "XDG_CONFIG_HOME")}
    env.update(SNAP=str(_snap(tmp_path, fmt)), FT=str(stub), GPUS=gpus, CGROUP="0", MEMGUARD="0",
               MIN_AVAIL_GIB="0", PORT="1", XDG_CONFIG_HOME=str(tmp_path / "config"))
    env.update(env_extra or {})
    r = subprocess.run(["bash", str(SCRIPTS / "serve-qwen38.sh")], env=env, capture_output=True,
                       text=True, timeout=60)
    got = dict(line.split("=", 1) for line in out.read_text().split())
    return got, r.stderr


def _write_calibration(tmp_path, gpus_key, fetch, threads):
    """The env file the calibrator writes (its writer, fed a minimal result)."""
    hw = {"date": "2026-10-28T20:00", "gpus_arg": gpus_key.replace("-", ","),
          "gpus": [{"name": "GPU", "pcie_gen": "1", "pcie_gen_max": "4", "pcie_width": "8", "h2d_gbps": 13.4}],
          "cpu": {"model": "CPU", "physical_cores": 8}, "cpu_kernel": {}}
    res = {"hw": hw, "depths": {"short": 4096}, "confirm": {"base": {"short": 35.0}, "shares": {fetch: {"short": 56.0}}},
           "decode_fetch": fetch, "threads": threads, "why": "test"}
    path = tmp_path / "config" / "freetoken" / f"calibration-gpus-{gpus_key}.env"
    cal.write_outputs(res, "report", path)
    return path


def test_serve_reads_the_calibration_of_its_gpu_set(tmp_path):
    _write_calibration(tmp_path, "0", 0.15, 6)
    _write_calibration(tmp_path, "0-1", 1.0, 4)
    got, err = _serve(tmp_path, "0")
    assert got == {"FREETOKEN_MOE_DECODE_FETCH": "0.15", "FREETOKEN_CPU_DECODE_THREADS": "6"}
    assert "DECODE_FETCH=0.15 CPU_DECODE_THREADS=6 (calibration " in err
    got, _ = _serve(tmp_path, "0,1")
    assert got == {"FREETOKEN_MOE_DECODE_FETCH": "1", "FREETOKEN_CPU_DECODE_THREADS": "4"}


def test_explicit_env_overrides_the_calibration(tmp_path):
    _write_calibration(tmp_path, "0", 0.15, 6)
    got, err = _serve(tmp_path, "0", {"DECODE_FETCH": "0.3"})
    assert got == {"FREETOKEN_MOE_DECODE_FETCH": "0.3"} and "(env)" in err
    got, _ = _serve(tmp_path, "0", {"CPU_DECODE_THREADS": "3"})
    assert got == {"FREETOKEN_MOE_DECODE_FETCH": "0.15", "FREETOKEN_CPU_DECODE_THREADS": "3"}


@pytest.mark.parametrize("gpus,want", [("0", "0.1"), ("0,1", "1")])
def test_defaults_without_a_calibration(tmp_path, gpus, want):
    got, err = _serve(tmp_path, gpus)
    assert got == {"FREETOKEN_MOE_DECODE_FETCH": want} and "(default)" in err


def test_int4_ignores_the_calibration(tmp_path):
    _write_calibration(tmp_path, "0", 0.15, 6)
    got, err = _serve(tmp_path, "0", fmt="w4a16")
    assert got == {"FREETOKEN_MOE_DECODE_FETCH": "1"} and "DECODE_FETCH" not in err


def test_calibration_file_is_parsed_not_sourced(tmp_path):
    path = tmp_path / "config" / "freetoken" / "calibration-gpus-0.env"
    path.parent.mkdir(parents=True)
    canary = tmp_path / "canary"
    path.write_text(f"DECODE_FETCH=$(touch {canary})\nCPU_DECODE_THREADS=6; touch {canary}\n")
    got, err = _serve(tmp_path, "0")
    assert not canary.exists()
    assert got == {"FREETOKEN_MOE_DECODE_FETCH": "0.1"} and "(default)" in err


def test_report_and_env_from_a_full_result(tmp_path):
    hw = {"date": "2026-10-28T20:00", "host": "box", "gpus_arg": "0",
          "gpus": [{"index": 0, "name": "GPU", "memory_mib": 20480, "pcie_gen": "4", "pcie_width": "8",
                    "pcie_gen_max": "4", "pcie_width_max": "16", "h2d_gbps": 13.4}],
          "cpu": {"model": "CPU", "physical_cores": 8, "logical_cpus": 16, "ram_gib": 91.8, "avx2": True,
                  "avx512f": True, "avx512_bf16": True, "avx512_vnni": True, "avx_vnni": True},
          "cpu_kernel": {"isa": "avx512f", "bytes_per_expert": 2_780_000,
                         "threads": {"7": {"us_per_task": {"1": 55.0, "2": 107.0, "8": 430.0},
                                           "us_per_expert": 53.8, "gbps": 51.7}}},
          "pcie_us_per_expert": 208.0, "default_threads": 7}
    depths = {"short": 4184, "long": 100371}
    screen = {1.0: {"short": 35.3, "long": 31.2}, 0.5: {"short": 43.2, "long": 39.5},
              0.15: {"short": 55.6, "long": 52.3}, 0.1: {"short": 56.0, "long": 52.0}}
    res = {"hw": hw, "depths": depths, "screen": screen, "threads_table": {3: 46.5, 7: 52.3, 8: 52.0},
           "confirm": {"base": screen[1.0], "shares": {0.15: {"short": 55.0, "long": 51.9},
                                                       0.1: {"short": 55.4, "long": 51.5}}},
           "decode_fetch": 0.15, "threads": 7, "why": "confirmation runs: 0.15 is +62%",
           "screen_threads": 7, "sweep_s": 600, "out_tokens": 192, "skip": 48}
    report = cal.render_report(res, 5)
    assert "Recommendation: DECODE_FETCH=0.15 CPU_DECODE_THREADS=7" in report and "PCIe gen4 x8" in report
    assert "0.15            55.0            51.9  <- pick" in report
    env = tmp_path / "calibration-gpus-0.env"
    cal.write_outputs(res, report, env)
    text = env.read_text()
    assert text.endswith("DECODE_FETCH=0.15\nCPU_DECODE_THREADS=7\n")
    assert "DECODE_FETCH=1 35.3 / 31.2, picked 55.0 / 51.9" in text and "54 us/expert at 7 threads" in text
