from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample(path: str | Path = "sample_gnss_data.csv", seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    path = Path(path)
    sats = ["G05", "G12", "G17", "G21", "G24", "G30"]
    times = np.arange(0, 301, 1.0)
    rows = []
    attack_start = 110
    pull_off_start = 160
    for sat in sats:
        cn0_base = rng.normal(43, 1.0)
        pr0 = rng.uniform(2.05e7, 2.35e7)
        dop0 = rng.normal(-1000, 200)
        for t in times:
            label = 1 if t >= attack_start else 0
            # Nominal signals.
            cn0 = cn0_base + rng.normal(0, 0.4)
            pseudorange = pr0 + 80 * t + rng.normal(0, 1.5)
            doppler = dop0 + rng.normal(0, 1.0)
            prompt = 1.0 + rng.normal(0, 0.025)
            early = 0.82 + rng.normal(0, 0.025)
            late = 0.81 + rng.normal(0, 0.025)

            # Spoofing process: embedding -> alignment -> pull-off -> separation.
            if t >= attack_start:
                cn0 += min(3.5, 0.02 * (t - attack_start)) + rng.normal(0, 0.2)
                prompt += 0.10 + rng.normal(0, 0.02)
                early += 0.08 + 0.0015 * (t - attack_start)
                late -= 0.04 + 0.0010 * (t - attack_start)
                doppler += 0.06 * (t - attack_start) + rng.normal(0, 0.5)
            if t >= pull_off_start:
                pseudorange += 0.30 * (t - pull_off_start) ** 2 + rng.normal(0, 5)
                doppler += 0.6 * (t - pull_off_start)
                early += 0.18
                late -= 0.10

            rows.append(
                {
                    "time": t,
                    "sat": sat,
                    "cn0": cn0,
                    "pseudorange": pseudorange,
                    "doppler": doppler,
                    "early": early,
                    "prompt": prompt,
                    "late": late,
                    "label": label,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


if __name__ == "__main__":
    print(generate_sample())
