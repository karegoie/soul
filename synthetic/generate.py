#!/usr/bin/env python3
import argparse
import random
from pathlib import Path


def mutate(seq: str, rate: float, rng: random.Random) -> str:
    bases = "ACGT"
    seq_list = list(seq)
    for i, b in enumerate(seq_list):
        if rng.random() < rate:
            seq_list[i] = rng.choice(bases.replace(b, ""))
    return "".join(seq_list)


def write_fasta(path: Path, name: str, seq: str) -> None:
    with path.open("w") as f:
        f.write(f">{name}\n")
        # wrap at 80 chars for readability
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")


def write_fastq(path: Path, reads) -> None:
    with path.open("w") as f:
        for name, seq in reads:
            f.write(f"@{name}\n{seq}\n+\n{'I'*len(seq)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic FASTA/FASTQ for testing.")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent, help="Output directory")
    parser.add_argument("--seed", type=int, default=1234, help="PRNG seed")
    parser.add_argument("--ref-len", type=int, default=5000, help="Reference length")
    parser.add_argument("--read-len", type=int, default=400, help="Read length")
    parser.add_argument("--step", type=int, default=100, help="Step size between read starts")
    parser.add_argument("--mut-rate", type=float, default=0.01, help="Per-base substitution rate")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    bases = "ACGT"
    ref = "".join(rng.choice(bases) for _ in range(args.ref_len))

    reads = []
    for start in range(0, args.ref_len - args.read_len + 1, args.step):
        seq = ref[start:start + args.read_len]
        mut = mutate(seq, args.mut_rate, rng)
        reads.append((f"read_{start}", mut))

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    write_fasta(outdir / "ref.fasta", "synthetic_ref", ref)
    write_fastq(outdir / "reads.fastq", reads)

    print(f"Wrote {len(reads)} reads of length {args.read_len} (step {args.step})")
    print(f"Reference length: {args.ref_len}, mutation rate: {args.mut_rate*100:.2f}%")
    print(f"Outputs: {outdir/'ref.fasta'}, {outdir/'reads.fastq'}")


if __name__ == "__main__":
    main()
