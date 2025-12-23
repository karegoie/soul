#!/usr/bin/env python3
import argparse
from pathlib import Path

def read_fasta(path: Path) -> str:
    seq = []
    with path.open() as f:
        for line in f:
            if line.startswith('>'):
                continue
            seq.append(line.strip())
    return ''.join(seq)


def read_fasta_multi(path: Path):
    contigs = []
    name = None
    buf = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if name is not None:
                    contigs.append((name, ''.join(buf)))
                name = line[1:]
                buf = []
            else:
                buf.append(line)
    if name is not None:
        contigs.append((name, ''.join(buf)))
    return contigs


def best_match(ref: str, query: str):
    best = None
    for start in range(len(ref) - len(query) + 1):
        window = ref[start:start+len(query)]
        mism = sum(a != b for a, b in zip(window, query))
        if best is None or mism < best[0]:
            best = (mism, start)
    return best


def main():
    p = argparse.ArgumentParser(description="Compare assembled contigs to synthetic reference")
    p.add_argument("contigs", type=Path, help="Assembled contig FASTA path")
    p.add_argument("--ref", type=Path, default=Path(__file__).parent / "ref.fasta", help="Reference FASTA")
    args = p.parse_args()

    contigs = read_fasta_multi(args.contigs)
    if not contigs:
        raise SystemExit("No contigs found in input FASTA")

    ref = read_fasta(args.ref)

    print(f"Reference length: {len(ref)}")
    print(f"Contigs: {len(contigs)}")
    print("name\tlen\tbest_start\tmismatches\tidentity")

    for name, seq in contigs:
        mism, start = best_match(ref, seq)
        identity = (len(seq) - mism) / len(seq)
        print(f"{name}\t{len(seq)}\t{start}\t{mism}\t{identity:.4f}")


if __name__ == "__main__":
    main()
