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
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input FASTQ file path
    #[arg(short, long)]
    input: PathBuf,

    /// Output GFA file path
    #[arg(short, long)]
    output: PathBuf,

    /// Number of threads to use
    #[arg(short, long, default_value_t = 8)]
    threads: usize,

    /// Minimum overlapping rhythm length (number of intervals)
    #[arg(short, long, default_value_t = 15)]
    min_rhythm_len: usize,
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

#[derive(Debug)]
struct ProcessedRead {
    id: usize,
    name: String,
    len: usize,
    // [Channel_Index][Strand] -> RhythmVector
    rhythms: Vec<HashMap<Strand, RhythmVector>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct IndexKey {
    channel_id: u8,
    d1: u32,
    d2: u32,
}

// ---------------------------------------------------------
// Logic: Permutations & Extraction
// ---------------------------------------------------------

fn generate_channels() -> Vec<Vec<u8>> {
    let bases = vec![b'A', b'C', b'G', b'T'];
    bases.into_iter().permutations(4).collect()
}

fn extract_rhythm_for_channel(seq: &[u8], kmer: &[u8]) -> RhythmVector {
    let mut positions = Vec::new();
    // Sliding window search
    for i in 0..seq.len().saturating_sub(kmer.len()) {
        if &seq[i..i + kmer.len()] == kmer {
            positions.push(i as u32);
        }
    }

    // Delta Encoding
    let mut distances = Vec::new();
    if positions.len() >= 2 {
        for w in positions.windows(2) {
            distances.push(w[1] - w[0]);
        }
    }
    distances
}

fn check_vector_overlap(vec_a: &[u32], vec_b: &[u32], min_len: usize) -> Option<usize> {
    if vec_a.len() < min_len || vec_b.len() < min_len {
        return None;
    }
    
    let max_check = std::cmp::min(vec_a.len(), vec_b.len());
    
    for len in (min_len..=max_check).rev() {
        let suffix_a = &vec_a[vec_a.len() - len..];
        let prefix_b = &vec_b[..len];
        
        if suffix_a == prefix_b {
            return Some(len);
        }
    }
    None
}

// ---------------------------------------------------------
// Main Execution
// ---------------------------------------------------------

fn main() -> Result<()> {
    // Initialize Logger (RUST_LOG environment variable controls level)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();
    
    info!("Starting Soul Assembler v0.1.0");
    info!("Configuration: Input={:?}, Threads={}, MinOverlap={}", args.input, args.threads, args.min_rhythm_len);

    rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global()?;

    // 1. Generate Channels
    let channels = generate_channels();
    info!("Generated {} rhythm channels (ACGT permutations)", channels.len());

    // 2. Load and Process Reads
    info!("Loading reads and extracting rhythm vectors...");
    let mut reader = parse_fastx_file(&args.input).context("Failed to open input file")?;
    let mut raw_records = Vec::new();
    
    while let Some(record) = reader.next() {
        let seqrec = record?;
        raw_records.push((
            String::from_utf8_lossy(seqrec.id()).to_string(),
            seqrec.seq().into_owned(),
        ));
    }

    if raw_records.is_empty() {
        warn!("No reads found in input file.");
        return Ok(());
    }

    let processed_reads: Vec<ProcessedRead> = raw_records
        .into_par_iter()
        .enumerate()
        .map(|(idx, (name, seq))| {
            let rc_seq = seq.reverse_complement();
            let mut read_rhythms = Vec::with_capacity(channels.len());

            for channel_kmer in &channels {
                let mut map = HashMap::new();
                map.insert(Strand::Forward, extract_rhythm_for_channel(&seq, channel_kmer));
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

    info!("Processed {} reads successfully.", processed_reads.len());

    // 3. Build Inverted Index
    info!("Building inverted index based on distance tuples...");
    let index = DashMap::new();

    processed_reads.par_iter().for_each(|read| {
        for (ch_idx, rhythm_map) in read.rhythms.iter().enumerate() {
            for (&strand, vec) in rhythm_map {
                if vec.len() < 2 { continue; }
                for (i, w) in vec.windows(2).enumerate() {
                    let key = IndexKey {
                        channel_id: ch_idx as u8,
                        d1: w[0],
                        d2: w[1],
                    };
                    index.entry(key).or_insert_with(Vec::new).push((read.id, strand, i));
                }
            }
        }
    });

    info!("Index construction complete. Total unique rhythm motifs: {}", index.len());

    // 4. Overlap Detection
    info!("Querying index and detecting overlaps...");
    let overlap_count = AtomicUsize::new(0);
    let edges = DashMap::new(); // Key: (ID_A, ID_B), Value: (Length, DirA, DirB)

    processed_reads.par_iter().for_each(|read_a| {
        let mut candidates: HashMap<(usize, Strand), usize> = HashMap::new();

        // Accumulate candidate hits
        for (ch_idx, rhythm_map) in read_a.rhythms.iter().enumerate() {
            let vec_a = &rhythm_map[&Strand::Forward];
            if vec_a.len() < 2 { continue; }

            for w in vec_a.windows(2) {
                let key = IndexKey {
                    channel_id: ch_idx as u8,
                    d1: w[0],
                    d2: w[1],
                };

                if let Some(hits) = index.get(&key) {
                    for &(read_b_id, strand_b, _) in hits.value() {
                        if read_a.id == read_b_id { continue; }
                        *candidates.entry((read_b_id, strand_b)).or_default() += 1;
                    }
                }
            }
        }

        // Verify candidates
        for ((read_b_id, strand_b), score) in candidates {
            // Filter weak candidates
            if score < 5 { continue; }
            
            // Deduplicate checks (only check if ID_A < ID_B to avoid double work)
            // Note: In a real assembler, we need to handle overlaps carefully.
            // Here we simply enforce an ordering to prevent duplicate processing.
            if read_a.id >= read_b_id { continue; }

            let read_b = &processed_reads[read_b_id];
            let mut max_overlap = 0;
            
            for ch_idx in 0..channels.len() {
                let vec_a = &read_a.rhythms[ch_idx][&Strand::Forward];
                let vec_b = &read_b.rhythms[ch_idx][&strand_b];

                if let Some(len) = check_vector_overlap(vec_a, vec_b, args.min_rhythm_len) {
                    // Approximate BP length: overlap_count * 256 (4^4)
                    let approx_bp = len * 256; 
                    if approx_bp > max_overlap {
                        max_overlap = approx_bp;
                    }
                }
            }

            if max_overlap > 0 {
                let dir_b = if strand_b == Strand::Forward { '+' } else { '-' };
                edges.insert((read_a.id, read_b_id), (max_overlap, '+', dir_b));
                overlap_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    });

    info!("Overlap detection complete. Found {} edges.", overlap_count.load(Ordering::Relaxed));

    // 5. Write GFA
    info!("Writing output to {:?}", args.output);
    let f = File::create(&args.output)?;
    let mut writer = BufWriter::new(f);
    
    writeln!(writer, "H\tVN:Z:1.0")?;

    // Segments
    for r in &processed_reads {
        writeln!(writer, "S\t{}\t*\tLN:i:{}", r.name, r.len)?;
    }

    // Links
    for r in edges.iter() {
        let (id_a, id_b) = r.key();
        let (len, dir_a, dir_b) = r.value();
        let name_a = &processed_reads[*id_a].name;
        let name_b = &processed_reads[*id_b].name;
        
        writeln!(writer, "L\t{}\t{}\t{}\t{}\t{}M", name_a, dir_a, name_b, dir_b, len)?;
    }

    info!("Soul completed successfully.");
    Ok(())
}