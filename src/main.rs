use anyhow::{Context, Result};
use clap::Parser;
use dashmap::DashMap;
use log::{debug, info};
use needletail::sequence::Sequence;
use pathfinding::prelude::astar;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use needletail::parse_fastx_file;

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
    #[arg(long, default_value_t = 1)]
    merge_match_bonus: isize,

    /// Merge scoring: mismatch penalty (per bp, positive number)
    #[arg(long, default_value_t = 2)]
    merge_mismatch_penalty: isize,

    /// Minimum identity for accepting an overlap merge (0.0-1.0)
    #[arg(long, default_value_t = 0.70)]
    merge_min_identity: f64,

    /// Allowed shift around the estimated overlap when scoring
    #[arg(long, default_value_t = 150)]
    merge_max_shift: usize,

    /// Minimum overlap length to consider during merge
    #[arg(long, default_value_t = 10)]
    merge_min_overlap: usize,

    /// Cap outgoing edges per node (keeps top overlaps)
    #[arg(long, default_value_t = 32)]
    max_edges_per_node: usize,
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

    let max_edges_per_node = args.max_edges_per_node;

    processed_reads.par_iter().for_each(|read_a| {
        let mut candidates = HashMap::new();
        // Index Query
        for (ch_idx, map) in read_a.rhythms.iter().enumerate() {
            let vec_a = &map[&Strand::Forward];
            if vec_a.len() < 2 { continue; }
            for w in vec_a.windows(2) {
                let key = IndexKey { channel_id: ch_idx as u16, d1: w[0], d2: w[1] };
                if let Some(hits) = index.get(&key) {
                    for &(read_b_id, strand_b, _) in hits.value() {
                        // Avoid self-matches
                        if read_a.id == read_b_id { continue; }
                        *candidates.entry((read_b_id, strand_b)).or_default() += 1;
                    }
                }
            }
        }

        // Step 4b: Verification & Alignment
        for ((read_b_id, strand_b), score) in candidates {
            // Filter weak candidates (heuristic threshold)
            if score < 5 { continue; }
            
            // Canonical ordering to avoid duplicate edges (A-B and B-A)
            if read_a.id >= read_b_id { continue; }

            let read_b = &processed_reads[read_b_id];
            let mut overlaps = Vec::new();
            
            // Check all channels to find the strongest support for this overlap
            for ch_idx in 0..channels.len() {
                let vec_a = &read_a.rhythms[ch_idx][&Strand::Forward];
                let vec_b = &read_b.rhythms[ch_idx][&strand_b];

                // Perform fuzzy matching logic
                if let Some((_rhythm_cnt, bp_len)) = check_vector_overlap_fuzzy(
                    vec_a, 
                    vec_b, 
                    args.min_rhythm_len, 
                    args.tolerance
                ) {
                    overlaps.push(bp_len);
                }
            }

            if overlaps.is_empty() { continue; }

            // Consensus Logic: Filter out noise by requiring multiple channels to agree
            overlaps.sort_unstable();

            let mut best_cluster_len = 0;
            let mut best_cluster_count = 0;
            
            let mut current_cluster_sum = 0;
            let mut current_cluster_count = 0;
            let mut current_cluster_start_val = overlaps[0];

            for &val in &overlaps {
                // Allow 5% difference relative to the cluster start or 50bp
                let diff = val.abs_diff(current_cluster_start_val);
                let threshold = (current_cluster_start_val / 20).max(50); 

                if diff <= threshold {
                    current_cluster_sum += val;
                    current_cluster_count += 1;
                } else {
                    // Check if this cluster is the best so far
                    if current_cluster_count > best_cluster_count {
                        best_cluster_count = current_cluster_count;
                        best_cluster_len = current_cluster_sum / current_cluster_count;
                    }
                    
                    // Start new cluster
                    current_cluster_start_val = val;
                    current_cluster_sum = val;
                    current_cluster_count = 1;
                }
            }
            
            // Check last cluster
            if current_cluster_count > best_cluster_count {
                best_cluster_count = current_cluster_count;
                best_cluster_len = current_cluster_sum / current_cluster_count;
            }

            // Require at least 3 channels to agree to accept the overlap
            if best_cluster_count >= 3 {
                // Determine orientation characters for GFA
                // Read A is always Forward (+).
                // If matched Strand B is Forward, it implies A+ overlaps B+.
                // If matched Strand B is Reverse, it implies A+ overlaps B-.
                let dir_b = if strand_b == Strand::Forward { '+' } else { '-' };
                
                edges.insert((read_a.id, read_b_id), (best_cluster_len, '+', dir_b));
                overlap_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    });

    info!("Overlap detection complete. Found {} valid edges.", overlap_count.load(Ordering::Relaxed));

    // 5. Write Output (GFA Format)
    info!("Writing GFA output to {:?}", args.output);
    let f = File::create(&args.output).context("Failed to create output file")?;
    let mut writer = BufWriter::new(f);

    writeln!(writer, "H\tVN:Z:1.0")?;

    // Write Segments (Nodes)
    for r in &processed_reads {
        // GFA Segment: S <Name> <Sequence>
        // We use '*' for sequence to keep GFA small, but actual seq can be added if needed.
        writeln!(writer, "S\t{}\t*\tLN:i:{}", r.name, r.len)?;
    }

    // Write Links (Edges)
    for r in edges.iter() {
        let (id_a, id_b) = r.key();
        let (len, dir_a, dir_b) = r.value();
        let name_a = &processed_reads[*id_a].name;
        let name_b = &processed_reads[*id_b].name;
        
        // GFA Link: L <SegA> <DirA> <SegB> <DirB> <CIGAR>
        // We use 'M' to denote the overlap length.
        writeln!(writer, "L\t{}\t{}\t{}\t{}\t{}M", name_a, dir_a, name_b, dir_b, len)?;
    }

    info!("Assembly complete: {} contigs, total {} bp -> {:?}", contigs.len(), total_bases, args.output);

    Ok(())
}