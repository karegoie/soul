use anyhow::{Context, Result};
use clap::Parser;
use dashmap::DashMap;
use log::{debug, info, warn};
use needletail::parse_fastx_file;
use needletail::sequence::Sequence;
use pathfinding::prelude::astar;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;

// ---------------------------------------------------------
// CLI Arguments
// ---------------------------------------------------------

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long)]
    output: PathBuf,

    #[arg(short, long, default_value_t = 8)]
    threads: usize,

    /// Number of "Rhythmers" (indexing channels) to select
    #[arg(long, default_value_t = 64)]
    num_rhythmers: usize,

    /// K-mer size for Rhythmer selection (13-17 recommended)
    #[arg(long, default_value_t = 15)]
    k: usize,

    /// Fuzzy matching tolerance (bp)
    #[arg(long, default_value_t = 5)]
    tolerance: u32,

    /// Merge scoring: match bonus (per bp)
    #[arg(long, default_value_t = 2)]
    merge_match_bonus: isize,

    /// Merge scoring: mismatch penalty (per bp, positive number)
    #[arg(long, default_value_t = 3)]
    merge_mismatch_penalty: isize,

    /// Minimum identity for accepting an overlap merge (0.0-1.0)
    #[arg(long, default_value_t = 0.85)]
    merge_min_identity: f64,

    /// Allowed shift around the estimated overlap when scoring
    #[arg(long, default_value_t = 150)]
    merge_max_shift: usize,

    /// Minimum overlap length to consider during merge
    #[arg(long, default_value_t = 60)]
    merge_min_overlap: usize,
}

// ---------------------------------------------------------
// Data Structures
// ---------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Strand {
    Forward,
    Reverse,
}

type RhythmVector = Vec<u32>;

struct MergeParams {
    match_bonus: isize,
    mismatch_penalty: isize,
    min_identity: f64,
    max_shift: usize,
    min_overlap: usize,
}

fn sample_variance(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return f64::MAX;
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let sum_sq: f64 = values.iter().map(|v| (v - mean).powi(2)).sum();
    sum_sq / (n as f64 - 1.0)
}

#[derive(Debug, Clone)]
struct GenomicRead {
    id: usize,
    #[allow(dead_code)]
    name: String,
    seq: Vec<u8>, // Keep sequence for Consensus step
}

struct ProcessedRead {
    id: usize,
    // [Channel_Index][Strand] -> RhythmVector
    rhythms: Vec<HashMap<Strand, RhythmVector>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct IndexKey {
    channel_id: u16,
    d1: u32,
    d2: u32,
}

/// Represents an edge in the Overlap Graph
#[derive(Clone, Debug, Eq, PartialEq)]
struct OverlapEdge {
    target_id: usize,
    overlap_len: usize,
    score: usize,
}

// ---------------------------------------------------------
// 1. Rhythmer Selection (The "Recurrentizer" Logic)
// ---------------------------------------------------------

/// Calculates distance variance for k-mers to find the most "regular" ones.
fn select_rhythmers(
    reads: &[GenomicRead],
    k: usize,
    top_n: usize,
) -> Vec<Vec<u8>> {
    info!("Phase 1: Rhythmer Selection (Scanning for regular patterns...)");
    
    // 1. Map k-mer -> List of Interval Distances
    // We use a subset of reads to save time if dataset is huge
    let sample_size = std::cmp::min(reads.len(), 5000);
    let sample_reads = &reads[..sample_size];

    let kmer_stats = DashMap::new(); // Parallel collection

    sample_reads.par_iter().for_each(|read| {
        let mut last_pos: HashMap<Vec<u8>, u32> = HashMap::new();
        
        for (i, win) in read.seq.windows(k).enumerate() {
            let kmer = win.to_vec();
            if let Some(prev_i) = last_pos.get(&kmer) {
                let dist = (i as u32) - prev_i;
                kmer_stats.entry(kmer.clone())
                    .or_insert_with(Vec::new)
                    .push(dist as f64);
            }
            last_pos.insert(kmer, i as u32);
        }
    });

    info!("Analyzed unique k-mers. Calculating variance...");

    // 2. Calculate Variance & Select
    let mut candidates: Vec<(Vec<u8>, f64, usize)> = kmer_stats
        .into_iter()
        .map(|(kmer, dists)| {
            let count = dists.len();
            if count < 5 { return (kmer, f64::MAX, count); } // Ignore rare kmers
            let variance = sample_variance(&dists);
            (kmer, variance, count)
        })
        .filter(|(_, var, _)| *var >= 0.0) // Filter invalid
        .collect();

    // Sort by Variance (Lower is more regular/rhythmic)
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    // Log top candidates
    for i in 0..std::cmp::min(5, candidates.len()) {
        debug!("Top Rhythmer #{}: {:?} (Var: {:.2}, Count: {})", 
            i, String::from_utf8_lossy(&candidates[i].0), candidates[i].1, candidates[i].2);
    }

    candidates.into_iter()
        .take(top_n)
        .map(|(kmer, _, _)| kmer)
        .collect()
}

// ---------------------------------------------------------
// 2. Indexing & Overlap (The "Soul" Logic)
// ---------------------------------------------------------

fn extract_rhythm(seq: &[u8], kmer: &[u8]) -> RhythmVector {
    let mut positions = Vec::new();
    for i in 0..seq.len().saturating_sub(kmer.len()) {
        if &seq[i..i + kmer.len()] == kmer {
            positions.push(i as u32);
        }
    }
    let mut dists = Vec::new();
    if positions.len() >= 2 {
        for w in positions.windows(2) {
            dists.push(w[1] - w[0]);
        }
    }
    dists
}

fn check_fuzzy_overlap(
    vec_a: &[u32], 
    vec_b: &[u32], 
    min_len: usize, 
    tolerance: u32
) -> Option<usize> {
    if vec_a.len() < min_len || vec_b.len() < min_len { return None; }
    let max_check = std::cmp::min(vec_a.len(), vec_b.len());

    // Check from longest overlap down
    'outer: for len in (min_len..=max_check).rev() {
        let suffix_a = &vec_a[vec_a.len() - len..];
        let prefix_b = &vec_b[..len];

        // Mean distance calculation for precise physical length
        let mut total_bp = 0;
        for (da, db) in suffix_a.iter().zip(prefix_b.iter()) {
            if da.abs_diff(*db) > tolerance { continue 'outer; }
            total_bp += (da + db) / 2;
        }
        return Some(total_bp as usize);
    }
    None
}

// ---------------------------------------------------------
// 3. Consensus (Block Aligner)
// ---------------------------------------------------------

/// Aligns two overlapping sequences and merges them.
/// Returns the consensus sequence.
fn align_and_merge(
    seq_a: &[u8], 
    seq_b: &[u8], 
    approx_overlap: usize,
    params: &MergeParams,
) -> Result<Vec<u8>> {
    let max_shift = params.max_shift;
    let mut best: Option<(isize, usize, usize, usize)> = None; // (score, ovlp, matches, mismatches)

    let min_ovlp = approx_overlap.saturating_sub(max_shift).max(params.min_overlap);
    let max_ovlp = approx_overlap + max_shift;

    for ovlp in min_ovlp..=max_ovlp {
        let ovlp = ovlp.min(seq_a.len()).min(seq_b.len());
        if ovlp == 0 {
            continue;
        }

        let start_a = seq_a.len() - ovlp;
        let (mut matches, mut mismatches) = (0usize, 0usize);
        for (a, b) in seq_a[start_a..].iter().zip(&seq_b[..ovlp]) {
            if a == b {
                matches += 1;
            } else {
                mismatches += 1;
            }
        }

        let score = params.match_bonus * matches as isize - params.mismatch_penalty * mismatches as isize;

        if let Some((best_score, best_ovlp, _, _)) = best {
            if score > best_score || (score == best_score && ovlp > best_ovlp) {
                best = Some((score, ovlp, matches, mismatches));
            }
        } else {
            best = Some((score, ovlp, matches, mismatches));
        }
    }

    let (_score, overlap_len, matches, _mismatches) = best.ok_or_else(|| anyhow::anyhow!("No overlap found"))?;
    if overlap_len == 0 {
        return Err(anyhow::anyhow!("Zero-length overlap"));
    }

    let identity = matches as f64 / overlap_len as f64;
    if identity < params.min_identity { // reject very noisy overlaps
        return Err(anyhow::anyhow!("Low-identity overlap"));
    }

    let mut merged = seq_a.to_vec();
    if overlap_len < seq_b.len() {
        merged.extend_from_slice(&seq_b[overlap_len..]);
    }

    Ok(merged)
}

// ---------------------------------------------------------
// Main Pipeline
// ---------------------------------------------------------

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    
    rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global()?;
    
    info!("Soul v0.3: Starting Rhythmer-based OLC Assembly");

    // ------------------------------------------------------
    // Step 0: Load Reads
    // ------------------------------------------------------
    let mut reader = parse_fastx_file(&args.input).context("Input read error")?;
    let mut reads = Vec::new();
    while let Some(record) = reader.next() {
        let r = record?;
        reads.push(GenomicRead {
            id: reads.len(),
            name: String::from_utf8_lossy(r.id()).to_string(),
            seq: r.seq().into_owned(),
        });
    }
    info!("Loaded {} reads.", reads.len());

    // ------------------------------------------------------
    // Step 1: Select Rhythmers (Data-Driven Channels)
    // ------------------------------------------------------
    let rhythmer_kmers = select_rhythmers(&reads, args.k, args.num_rhythmers);
    info!("Selected {} optimal Rhythmers.", rhythmer_kmers.len());

    // ------------------------------------------------------
    // Step 2: Extract Rhythms & Build Index
    // ------------------------------------------------------
    info!("Phase 2: Indexing Rhythms...");
    let processed_reads: Vec<ProcessedRead> = reads.par_iter()
        .map(|read| {
            let rc_seq = read.seq.reverse_complement();
            let mut rhythms = Vec::new();
            
            for kmer in &rhythmer_kmers {
                let mut map = HashMap::new();
                map.insert(Strand::Forward, extract_rhythm(&read.seq, kmer));
                map.insert(Strand::Reverse, extract_rhythm(&rc_seq, kmer));
                rhythms.push(map);
            }
            ProcessedRead { id: read.id, rhythms }
        })
        .collect();

    let index = DashMap::new();
    processed_reads.par_iter().for_each(|pr| {
        for (ch_idx, map) in pr.rhythms.iter().enumerate() {
            for (&strand, vec) in map {
                if vec.len() < 2 { continue; }
                for (i, w) in vec.windows(2).enumerate() {
                    let key = IndexKey { channel_id: ch_idx as u16, d1: w[0], d2: w[1] };
                    index.entry(key).or_insert_with(Vec::new).push((pr.id, strand, i));
                }
            }
        }
    });
    info!("Index built. Size: {}", index.len());

    // ------------------------------------------------------
    // Step 3: Build Overlap Graph
    // ------------------------------------------------------
    info!("Phase 3: Building Overlap Graph...");
    // Adjacency List: NodeID -> List of Edges
    let graph = Arc::new(DashMap::new());

    processed_reads.par_iter().for_each(|read_a| {
        let mut candidates = HashMap::new();
        // Index Query
        for (ch_idx, map) in read_a.rhythms.iter().enumerate() {
            let vec_a = &map[&Strand::Forward];
            if vec_a.len() < 2 { continue; }
            for w in vec_a.windows(2) {
                let key = IndexKey { channel_id: ch_idx as u16, d1: w[0], d2: w[1] };
                if let Some(hits) = index.get(&key) {
                    for &(bid, strand_b, _) in hits.value() {
                        if read_a.id != bid {
                            *candidates.entry((bid, strand_b)).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        // Verify & Add Edges
        for ((bid, strand_b), score) in candidates {
            if score < 1 { continue; } // Yeast-scale: allow more edges, rely on overlap/identity filters later
            let read_b = &processed_reads[bid];
            
            let mut best_overlap = 0;
            for ch in 0..rhythmer_kmers.len() {
                if let Some(len) = check_fuzzy_overlap(
                    &read_a.rhythms[ch][&Strand::Forward],
                    &read_b.rhythms[ch][&strand_b],
                    3, // Moderate interval requirement
                    args.tolerance
                ) {
                    if len > best_overlap { best_overlap = len; }
                }
            }

            if best_overlap > 0 {
                // Add directed edge
                graph.entry(read_a.id).or_insert_with(Vec::new).push(OverlapEdge {
                    target_id: bid,
                    overlap_len: best_overlap,
                    score,
                });
            }
        }
    });

    // ------------------------------------------------------
    // Step 4: Layout (A* / Greedy Extension)
    // ------------------------------------------------------
    info!("Phase 4: Layout & Pathfinding...");
    info!("Graph nodes: {}", graph.len());
    let merge_params = MergeParams {
        match_bonus: args.merge_match_bonus,
        mismatch_penalty: args.merge_mismatch_penalty,
        min_identity: args.merge_min_identity,
        max_shift: args.merge_max_shift,
        min_overlap: args.merge_min_overlap,
    };

    let mut visited = HashSet::new();
    let mut contig_idx = 0usize;
    let mut total_bases = 0usize;
    let mut output = BufWriter::new(File::create(&args.output)?);

    while visited.len() < reads.len() {
        // Pick next seed: longest unvisited read; prefer one present in the graph
        let seed_opt = reads
            .iter()
            .enumerate()
            .filter(|(id, _)| !visited.contains(id))
            .max_by_key(|(id, r)| {
                let in_graph = graph.contains_key(id);
                (in_graph as usize, r.seq.len())
            });

        let Some((seed_id, seed_read)) = seed_opt else { break; };

        let max_read_len = seed_read.seq.len() * 2; // cost inversion base

        let result = astar(
            &seed_id,
            |&node_id| {
                if let Some(entry) = graph.get(&node_id) {
                    entry
                        .iter()
                        .filter(|e| !visited.contains(&e.target_id))
                        .map(|e| {
                            let cost = max_read_len.saturating_sub(e.overlap_len) as i32;
                            (e.target_id, cost)
                        })
                        .collect()
                } else {
                    Vec::new()
                }
            },
            |&_node_id| 0, // heuristic
            |&node_id| {
                let has_unvisited = graph
                    .get(&node_id)
                    .map(|edges| edges.iter().any(|e| !visited.contains(&e.target_id)))
                    .unwrap_or(false);
                !has_unvisited
            },
        );

        let path: Vec<usize> = if let Some((p, _)) = result { p } else { vec![seed_id] };

        // Consensus & Output for this contig
        let mut contig_seq = reads[path[0]].seq.clone();

        for win in path.windows(2) {
            let prev = &reads[win[0]];
            let curr = &reads[win[1]];

            // Retrieve overlap info from graph
            let edges = graph.get(&win[0]).unwrap();
            let edge = edges.iter().find(|e| e.target_id == win[1]).unwrap();

            match align_and_merge(&contig_seq, &curr.seq, edge.overlap_len, &merge_params) {
                Ok(merged) => contig_seq = merged,
                Err(e) => warn!("Alignment failed between {} and {}: {}", prev.id, curr.id, e),
            }
        }

        contig_idx += 1;
        total_bases += contig_seq.len();
        writeln!(output, ">Soul_Contig_{} length={}", contig_idx, contig_seq.len())?;
        output.write_all(&contig_seq)?;
        output.write_all(b"\n")?;

        for nid in path {
            visited.insert(nid);
        }
    }

    info!("Assembly complete: {} contigs, total {} bp -> {:?}", contig_idx, total_bases, args.output);

    Ok(())
}