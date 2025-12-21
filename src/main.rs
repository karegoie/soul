use anyhow::{Context, Result};
use clap::Parser;
use dashmap::DashMap;
use itertools::Itertools;
use log::{info, warn};
use needletail::parse_fastx_file;
use needletail::sequence::Sequence;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Soul: T2T Haplotype Phased Assembler
/// 
/// A proof-of-concept assembler that uses "Rhythm" (inter-kmer distances)
/// to resolve repetitive regions and phase haplotypes.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input FASTQ file path (HiFi reads recommended)
    #[arg(short, long)]
    input: PathBuf,

    /// Output GFA file path
    #[arg(short, long)]
    output: PathBuf,

    /// Number of threads to use for parallel processing
    #[arg(short, long, default_value_t = 8)]
    threads: usize,

    /// Minimum overlapping rhythm length (number of intervals, not bp)
    #[arg(short, long, default_value_t = 15)]
    min_rhythm_len: usize,

    /// Fuzzy matching tolerance in bp.
    /// Allows small differences between intervals (e.g., indels).
    /// If tolerance is 2, distance 2000 matches 1998-2002.
    #[arg(short, long, default_value_t = 2)]
    tolerance: u32,
}

// ---------------------------------------------------------
// Data Structures
// ---------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Strand {
    Forward,
    Reverse,
}

/// A "Rhythm" is a sequence of distances between specific k-mers.
/// We use u32 to save memory (supports distances up to ~4Gb).
type RhythmVector = Vec<u32>;

/// Holds the rhythm profiles for a single read.
#[derive(Debug)]
struct ProcessedRead {
    id: usize,
    name: String,
    len: usize,
    // Maps: [Channel_Index] -> [Strand] -> RhythmVector
    // We store pre-calculated rhythms for both Forward and Reverse strands
    // to enable efficient all-vs-all comparison.
    rhythms: Vec<HashMap<Strand, RhythmVector>>,
}

/// The Inverted Index Key: (Channel_ID, Distance_1, Distance_2)
/// We index pairs of distances to create a more unique signature than single distances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct IndexKey {
    channel_id: u8,
    d1: u32,
    d2: u32,
}

// ---------------------------------------------------------
// Core Logic: Rhythm Extraction & Comparison
// ---------------------------------------------------------

/// Generate all 24 permutations of "ACGT" (4-mers) to serve as anchor channels.
fn generate_channels() -> Vec<Vec<u8>> {
    let bases = vec![b'A', b'C', b'G', b'T'];
    bases.into_iter().permutations(4).collect()
}

/// Scans a sequence for a specific k-mer and calculates distances between occurrences.
fn extract_rhythm_for_channel(seq: &[u8], kmer: &[u8]) -> RhythmVector {
    let mut positions = Vec::new();
    
    // Naive sliding window search.
    // For production, SIMD or Aho-Corasick would be faster.
    for i in 0..seq.len().saturating_sub(kmer.len()) {
        if &seq[i..i + kmer.len()] == kmer {
            positions.push(i as u32);
        }
    }

    // Convert absolute positions to relative delta distances.
    // e.g., Positions: [100, 300, 305] -> Distances: [200, 5]
    let mut distances = Vec::new();
    if positions.len() >= 2 {
        for w in positions.windows(2) {
            distances.push(w[1] - w[0]);
        }
    }
    distances
}

/// Checks if the suffix of Vector A matches the prefix of Vector B.
/// Uses fuzzy logic to tolerate small indels.
/// 
/// Returns: Option<(Matched_Interval_Count, Physical_Overlap_Length_BP)>
fn check_vector_overlap_fuzzy(
    vec_a: &[u32], 
    vec_b: &[u32], 
    min_len: usize,
    tolerance: u32
) -> Option<(usize, usize)> {
    if vec_a.len() < min_len || vec_b.len() < min_len {
        return None;
    }
    
    // Determine the maximum possible overlap length in intervals
    let max_check = std::cmp::min(vec_a.len(), vec_b.len());
    
    // Iterate from longest possible overlap down to minimum required length
    'outer: for len in (min_len..=max_check).rev() {
        let suffix_a = &vec_a[vec_a.len() - len..];
        let prefix_b = &vec_b[..len];
        
        // Element-wise fuzzy comparison
        for (da, db) in suffix_a.iter().zip(prefix_b.iter()) {
            if da.abs_diff(*db) > tolerance {
                continue 'outer; // Mismatch found, try next length
            }
        }

        // If we reach here, the vectors match within tolerance.
        // Calculate the exact physical distance by summing the intervals.
        // We use the average of A and B for each interval to smooth out indels.
        let overlap_bp: u32 = suffix_a.iter()
            .zip(prefix_b.iter())
            .map(|(a, b)| (a + b) / 2) 
            .sum();

        return Some((len, overlap_bp as usize));
    }
    None
}

// ---------------------------------------------------------
// Main Execution Flow
// ---------------------------------------------------------

fn main() -> Result<()> {
    // Initialize logger (default level: info)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();
    
    info!("Starting Soul Assembler v0.2.0");
    info!("Config: Input={:?}, Threads={}, MinRhythm={}, Tolerance={}", 
        args.input, args.threads, args.min_rhythm_len, args.tolerance);

    // Setup thread pool
    rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global()?;

    // 1. Generate Channels
    // We use 24 permutations of ACGT to create a "barcode" for the genome.
    let channels = generate_channels();
    info!("Generated {} rhythm channels (ACGT permutations)", channels.len());

    // 2. Load Reads and Extract Rhythms
    info!("Loading reads and extracting rhythm vectors...");
    let mut reader = parse_fastx_file(&args.input).context("Failed to open input FASTQ")?;
    let mut raw_records = Vec::new();
    
    while let Some(record) = reader.next() {
        let seqrec = record?;
        // We must own the data to process in parallel later
        raw_records.push((
            String::from_utf8_lossy(seqrec.id()).to_string(),
            seqrec.seq().into_owned(),
        ));
    }

    if raw_records.is_empty() {
        warn!("Input file is empty. Exiting.");
        return Ok(());
    }

    // Parallel processing of reads
    let processed_reads: Vec<ProcessedRead> = raw_records
        .into_par_iter()
        .enumerate()
        .map(|(idx, (name, seq))| {
            let rc_seq = seq.reverse_complement();
            let mut read_rhythms = Vec::with_capacity(channels.len());

            for channel_kmer in &channels {
                let mut map = HashMap::new();
                // Extract rhythm for Forward strand
                map.insert(Strand::Forward, extract_rhythm_for_channel(&seq, channel_kmer));
                // Extract rhythm for Reverse strand (essential for detecting RC overlaps)
                map.insert(Strand::Reverse, extract_rhythm_for_channel(&rc_seq, channel_kmer));
                read_rhythms.push(map);
            }

            ProcessedRead {
                id: idx,
                name,
                len: seq.len(),
                rhythms: read_rhythms,
            }
        })
        .collect();

    info!("Successfully processed {} reads.", processed_reads.len());

    // 3. Build Inverted Index
    // We index consecutive distance pairs (d1, d2) to quickly find candidate overlaps.
    info!("Building inverted index based on distance tuples...");
    
    // Key: (Channel, Dist1, Dist2), Value: List of (ReadID, Strand, Offset)
    let index = DashMap::new();

    processed_reads.par_iter().for_each(|read| {
        for (ch_idx, rhythm_map) in read.rhythms.iter().enumerate() {
            for (&strand, vec) in rhythm_map {
                if vec.len() < 2 { continue; }
                
                // Sliding window of size 2 over the rhythm vector
                for (i, w) in vec.windows(2).enumerate() {
                    let key = IndexKey {
                        channel_id: ch_idx as u8,
                        d1: w[0],
                        d2: w[1],
                    };
                    // Insert into concurrent hashmap
                    index.entry(key).or_insert_with(Vec::new).push((read.id, strand, i));
                }
            }
        }
    });

    info!("Index construction complete. Unique rhythm motifs: {}", index.len());

    // 4. Overlap Detection
    info!("Querying index and detecting overlaps...");
    
    let overlap_count = AtomicUsize::new(0);
    // Stores edges: Key=(ReadA_ID, ReadB_ID), Value=(Overlap_BP, DirA, DirB)
    let edges = DashMap::new();

    processed_reads.par_iter().for_each(|read_a| {
        let mut candidates: HashMap<(usize, Strand), usize> = HashMap::new();

        // Step 4a: Candidate Search
        // We only use the Forward strand of Read A to query the index.
        for (ch_idx, rhythm_map) in read_a.rhythms.iter().enumerate() {
            let vec_a = &rhythm_map[&Strand::Forward];
            if vec_a.len() < 2 { continue; }

            for w in vec_a.windows(2) {
                // Exact match lookup for seeds (speed optimization)
                let key = IndexKey {
                    channel_id: ch_idx as u8,
                    d1: w[0],
                    d2: w[1],
                };

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
            let mut max_bp_len = 0;
            
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
                    if bp_len > max_bp_len {
                        max_bp_len = bp_len;
                    }
                }
            }

            if max_bp_len > 0 {
                // Determine orientation characters for GFA
                // Read A is always Forward (+).
                // If matched Strand B is Forward, it implies A+ overlaps B+.
                // If matched Strand B is Reverse, it implies A+ overlaps B-.
                let dir_b = if strand_b == Strand::Forward { '+' } else { '-' };
                
                edges.insert((read_a.id, read_b_id), (max_bp_len, '+', dir_b));
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

    info!("Soul pipeline completed successfully.");
    Ok(())
}