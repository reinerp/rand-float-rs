#!/usr/bin/env python3
"""Plot the output of `cargo bench` as a grouped bar chart.

Parses Criterion's terminal output (given as a file argument, or piped on
standard input) and draws, for every technique, three side-by-side bars in ns
per generated f64 (lower is better): the time per call of the `per_call` group,
the same minus the bare-source baseline (i.e. the cost of the conversion alone,
which is meaningful in the serial single-call setting, where source and
conversion costs compose roughly additively), and the per-element time of the
the `fill_1024` and `sum_1024` groups (not baseline-subtracted: in vectorized
loops the source is fused with the conversion, so the standalone baseline is
not a valid subtrahend).

Typical use:

    cargo bench 2>&1 | python3 python/plot_bench.py -o bench.pdf
"""

import argparse
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIME_TO_NS = {"ps": 1e-3, "ns": 1.0, "µs": 1e3, "us": 1e3, "ms": 1e6}
THRPT_TO_NS_PER_ELEM = {"Kelem/s": 1e6, "Melem/s": 1e3, "Gelem/s": 1.0}

BENCH_RE = re.compile(r"\b(per_call|fill_1024|sum_1024)/(\S+)")
TIME_RE = re.compile(
    r"time:\s*\[\s*[\d.]+\s+\S+\s+([\d.]+)\s+(\S+)\s+[\d.]+\s+\S+\s*\]"
)
THRPT_RE = re.compile(
    r"thrpt:\s*\[\s*[\d.]+\s+\S+\s+([\d.]+)\s+(\S+)\s+[\d.]+\s+\S+\s*\]"
)


def parse(text):
    """Returns (order, single, fill, sum): the technique names in order of first
    appearance and each group's point estimates in ns per element."""
    order, single, fill, sum_ = [], {}, {}, {}
    current = None

    for line in text.splitlines():
        m = BENCH_RE.search(line)
        if m:
            current = (m.group(1), m.group(2))
            if m.group(2) not in order:
                order.append(m.group(2))
        # Criterion's comparison-with-previous-run lines also say "time:" /
        # "thrpt:", but with percentages; skip them.
        if "%" in line or current is None:
            continue
        group, name = current
        m = TIME_RE.search(line)
        if m and group == "per_call" and name not in single:
            single[name] = float(m.group(1)) * TIME_TO_NS[m.group(2)]
        m = THRPT_RE.search(line)
        if m and group == "fill_1024" and name not in fill:
            # Throughput per element, inverted into ns per element.
            fill[name] = THRPT_TO_NS_PER_ELEM[m.group(2)] / float(m.group(1))
        if m and group == "sum_1024" and name not in sum_:
            sum_[name] = THRPT_TO_NS_PER_ELEM[m.group(2)] / float(m.group(1))
    order = [n for n in order if n in single or n in fill or n in sum_]
    return order, single, fill, sum_


BASELINE = "weyl_baseline"


def plot(order, single, fill, sum_, output):
    x = range(len(order))
    width = 0.2

    fig, ax = plt.subplots(figsize=(2.0 * len(order) + 2, 5.5))

    bars_single = ax.bar(
        [i - 1.5 * width for i in x],
        [single.get(n, 0.0) for n in order],
        width,
        color="tab:blue",
        label="single call",
    )
    # Conversion-only cost: single call minus the bare-source baseline
    # (omitted for the baseline itself, and when no baseline was measured).
    base = single.get(BASELINE)
    bars_conv = ax.bar(
        [i - 0.5 * width for i in x],
        [
            single[n] - base
            if base is not None and n != BASELINE and n in single
            else 0.0
            for n in order
        ],
        width,
        color="tab:green",
        label="single call − baseline (conversion only)",
    )
    bars_fill = ax.bar(
        [i + 0.5 * width for i in x],
        [fill.get(n, 0.0) for n in order],
        width,
        color="tab:orange",
        label="fill of a 1024-element array",
    )
    bars_sum = ax.bar(
        [i + 1.5 * width for i in x],
        [sum_.get(n, 0.0) for n in order],
        width,
        color="tab:red",
        label="sum of 1024 generated values",
    )

    ax.set_ylabel("ns per generated f64 (lower is better)")
    ax.set_xticks(list(x), order, rotation=20, ha="right")
    ax.bar_label(bars_single, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(
        bars_conv,
        labels=[
            f"{r.get_height():.2f}" if r.get_height() != 0.0 else "" for r in bars_conv
        ],
        padding=2,
        fontsize=8,
    )
    ax.bar_label(bars_fill, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(bars_sum, fmt="%.2f", padding=2, fontsize=8)
    ax.set_title("u64 → f64 in [0 . . 1): conversion techniques")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    fig.tight_layout()
    # The format follows the file extension (.pdf, .png, .svg, ...).
    fig.savefig(output, dpi=150)
    print(f"wrote {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "input",
        nargs="?",
        help="file with the output of `cargo bench` (default: standard input)",
    )
    p.add_argument(
        "-o",
        "--output",
        default="bench.pdf",
        help="output image; the format follows the extension (.pdf, .png, .svg, ...)",
    )
    args = p.parse_args()

    text = open(args.input).read() if args.input else sys.stdin.read()
    order, single, fill, sum_ = parse(text)
    if not order:
        sys.exit("no benchmark results found in input")
    plot(order, single, fill, sum_, args.output)


if __name__ == "__main__":
    main()
