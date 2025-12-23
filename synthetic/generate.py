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

def write_reads_fasta(path: Path, reads) -> None:
    with path.open("w") as f:
        for name, seq in reads:
            f.write(f">{name}\n{seq}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic FASTA/FASTQ for testing.")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent, help="Output directory")
    parser.add_argument("--seed", type=int, default=1234, help="PRNG seed")
    parser.add_argument("--num-chromosomes", type=int, default=2, help="Number of chromosomes")
    parser.add_argument("--ref-len", type=int, default=10000, help="Reference length per chromosome")
    parser.add_argument("--read-len", type=int, default=1000, help="Read length")
    parser.add_argument("--coverage", type=int, default=30, help="Target coverage")
    parser.add_argument("--mut-rate", type=float, default=0.01, help="Per-base substitution rate")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    bases = "ACGT"
    
    refs = []
    for i in range(args.num_chromosomes):
        seq = "".join(rng.choice(bases) for _ in range(args.ref_len))
        refs.append((f"chr_{i+1}", seq))

    # Calculate number of reads needed for coverage
    # N * L = C * G  => N = (C * G) / L
    total_ref_len = args.ref_len * args.num_chromosomes
    num_reads = int((args.coverage * total_ref_len) / args.read_len)
    
    reads = []
    for i in range(num_reads):
        # Pick a random chromosome
        chr_name, ref_seq = rng.choice(refs)
        
        # Random start position
        if len(ref_seq) <= args.read_len:
            start = 0
            seq = ref_seq
        else:
            start = rng.randint(0, len(ref_seq) - args.read_len)
            seq = ref_seq[start:start + args.read_len]
        
        # 50% chance of reverse complement
        if rng.random() < 0.5:
            # Simple RC
            complement = {"A": "T", "C": "G", "G": "C", "T": "A"}
            seq = "".join(complement[b] for b in reversed(seq))
            strand = "-"
        else:
            strand = "+"

        mut = mutate(seq, args.mut_rate, rng)
        reads.append((f"read_{i}_{chr_name}_{strand}", mut))

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    
    with (outdir / "ref.fasta").open("w") as f:
        for name, seq in refs:
            f.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")
                
    write_fastq(outdir / "reads.fastq", reads)
    write_reads_fasta(outdir / "reads.fasta", reads)

    print(f"Wrote {len(reads)} reads of length {args.read_len} (Target Coverage {args.coverage}x)")
    print(f"Generated {args.num_chromosomes} chromosomes of length {args.ref_len}")
    print(f"Mutation rate: {args.mut_rate*100:.2f}%")
    print(f"Outputs: {outdir/'ref.fasta'}, {outdir/'reads.fastq'}, {outdir/'reads.fasta'}")


if __name__ == "__main__":
    main()
