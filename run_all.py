import argparse
import subprocess
import sys


def run(cmd):
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    args = parser.parse_args()

    base = [
        sys.executable,
        "-m",
        "text2python.train",
        "--epochs",
        str(args.epochs),
        "--train-size",
        str(args.train_size),
        "--val-size",
        str(args.val_size),
        "--test-size",
        str(args.test_size),
        "--device",
        args.device,
    ]

    for model in ["rnn", "lstm", "attention"]:
        run(base + ["--model", model])
        ckpt = f"checkpoints/{model}_best.pt"
        run(
            [
                sys.executable,
                "-m",
                "text2python.eval",
                "--checkpoint",
                ckpt,
                "--device",
                args.device,
            ]
        )
        if model == "attention":
            run(
                [
                    sys.executable,
                    "-m",
                    "text2python.attention",
                    "--checkpoint",
                    ckpt,
                    "--device",
                    args.device,
                    "--indices",
                    "0",
                    "1",
                    "2",
                ]
            )


if __name__ == "__main__":
    main()
