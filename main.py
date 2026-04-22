"""
CLI entry point.

Commands:
  python main.py train     -- fine-tune all candidate models
  python main.py benchmark -- evaluate all candidates, print Table I
  python main.py predict   -- interactive sentiment scoring + modulation demo
  python main.py demo      -- paper worked example (p=-0.7, s=+0.8)
"""
import argparse
import sys

from sentiment_trading.config import CANDIDATE_MODELS
from sentiment_trading.modulator import modulate, apply_decay
from sentiment_trading.pipeline import TradingPipeline
from sentiment_trading.selector import ModelSelector
from sentiment_trading.trainer import train_model


def cmd_train(args):
    for key, hf_name in CANDIDATE_MODELS.items():
        metrics = train_model(key, hf_name, force=args.force)
        print(
            f"{key:30s}  acc={metrics['accuracy']:.4f}  "
            f"f1={metrics['f1']:.4f}  mae={metrics['mae']:.4f}"
        )


def cmd_benchmark(args):
    selector = ModelSelector()
    results = selector.benchmark(force=args.force)
    selector.select()
    selector.print_table()

    # Print the 81.7pp gap highlighted in the paper
    accs = {k: v["accuracy"] for k, v in results.items()}
    if "deberta-finance" in accs and "pmatorras" in accs:
        gap = (accs["deberta-finance"] - accs["pmatorras"]) * 100
        print(f"\nBest–worst accuracy gap: {gap:.1f} percentage points")
        if accs["pmatorras"] < 0.333:
            print(
                f"pmatorras ({accs['pmatorras']*100:.1f}%) is below the "
                f"33.3% random baseline — automatic benchmarking is safety-critical."
            )


def cmd_predict(args):
    pipeline = TradingPipeline()
    pipeline.setup()
    print("\nEnter: <base_signal> | <news_text> | <dt_minutes>")
    print("Example: -0.7 | Apple beats earnings | 15")
    print("Ctrl-C to exit.\n")
    while True:
        try:
            raw = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) < 2:
            print("Need at least: base_signal | news_text")
            continue
        try:
            p = float(parts[0])
            text = parts[1]
            dt = float(parts[2]) if len(parts) > 2 else 0.0
        except ValueError:
            print("base_signal must be a number.")
            continue
        sig = pipeline.generate(p, text, dt)
        direction = "LONG" if sig.raw_price_signal > 0 else "SHORT"
        print(
            f"  [{direction}] base={sig.raw_price_signal:+.4f}  "
            f"s={sig.sentiment_score:+.4f}  "
            f"final={sig.final_signal:+.4f}  "
            f"(model={sig.model_used}, dt={sig.dt_minutes:.0f}min)"
        )


def cmd_demo(_args):
    """Paper Section III-F worked example: p=-0.7, s=+0.8."""
    p, s = -0.7, +0.8
    final = modulate(p, s)
    print("\n=== Paper Worked Example (Section III-F) ===")
    print(f"  p = {p} (short position)")
    print(f"  s = {s} (strongly positive news)")
    print(f"  sign(p)={'-' if p<0 else '+'}, sign(s)={'+' if s>0 else '-'}  → OPPOSED")
    print(f"  final = {p} * (1 - |{s}|) = {p} * {1-abs(s):.1f} = {final:.4f}")
    print(f"  Short conviction reduced from |{p}| to |{final:.4f}|")

    print("\n=== Temporal Decay Demo ===")
    for dt in [0, 30, 60, 90, 120]:
        s_decayed = apply_decay(s, dt)
        f = modulate(p, s_decayed)
        print(f"  dt={dt:3d}min  s_decayed={s_decayed:+.4f}  final={f:+.4f}")


COMMANDS = {
    "train": cmd_train,
    "benchmark": cmd_benchmark,
    "predict": cmd_predict,
    "demo": cmd_demo,
}


def main():
    parser = argparse.ArgumentParser(description="Sentiment-modulated trading pipeline")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="Fine-tune all candidate models")
    p_train.add_argument("--force", action="store_true", help="Re-train even if checkpoint exists")

    p_bench = sub.add_parser("benchmark", help="Benchmark all candidates (Table I)")
    p_bench.add_argument("--force", action="store_true", help="Re-run even if results cached")

    sub.add_parser("predict", help="Interactive prediction + modulation")
    sub.add_parser("demo", help="Paper Section III-F worked example")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
