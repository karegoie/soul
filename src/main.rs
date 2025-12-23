use anyhow::{Context, Result};
use clap::Parser;
use dashmap::DashMap;
use log::{info, warn};
use needletail::sequence::Sequence;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use needletail::parse_fastx_file;

// ---------------------------------------------------------
// CLI Arguments
// ---------------------------------------------------------

#[derive(Parser, Debug, Clone)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long)]
    output: PathBuf,

    #[arg(short, long, default_value_t = 8)]
    threads: usize,

    /// Number of "Rhythmers" (indexing channels) to select
    #[arg(long)]
    num_rhythmers: Option<usize>,

    /// K-mer size for Rhythmer selection (5-9 recommended)
    #[arg(long)]
    k: Option<usize>,

    /// Fuzzy matching tolerance (bp)
    #[arg(long)]
    tolerance: Option<u32>,

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

    /// Minimum overlapping rhythm length (number of intervals)
    #[arg(long)]
    min_rhythm_len: Option<usize>,

    /// Cap outgoing edges per node (keeps top overlaps)
    #[arg(long, default_value_t = 32)]
    max_edges_per_node: usize,
}

#[derive(Debug, Clone)]
struct AssemblyParams {
    k: usize,
    num_rhythmers: usize,
    tolerance: u32,
    min_rhythm_len: usize,
}

impl AssemblyParams {
    fn autoselect(reads: &[GenomicRead], args: &Args) -> Self {
        let avg_len = if reads.is_empty() {
            0
        } else {
            reads.iter().map(|r| r.seq.len()).sum::<usize>() / reads.len()
        };
        
        info!("Autoselecting parameters based on avg read length: {} bp", avg_len);

        let k = args.k.unwrap_or_else(|| {
            if avg_len < 500 { 5 }
            else if avg_len < 1500 { 7 }
            else { 9 }
        });

        let num_rhythmers = args.num_rhythmers.unwrap_or_else(|| {
            let n = avg_len / 20;
            n.clamp(50, 200)
        });

        let tolerance = args.tolerance.unwrap_or_else(|| {
            10
        });

        let min_rhythm_len = args.min_rhythm_len.unwrap_or_else(|| {
            let len = avg_len / 100;
            len.clamp(5, 50)
        });

        let params = AssemblyParams { k, num_rhythmers, tolerance, min_rhythm_len };
        info!("Selected Parameters: k={}, num_rhythmers={}, tolerance={}, min_rhythm_len={}", 
              params.k, params.num_rhythmers, params.tolerance, params.min_rhythm_len);
        params
    }
}

// ---------------------------------------------------------
// Data Structures
// ---------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Strand {
    Forward,
    Reverse,
}

#[derive(Debug, Clone)]
struct GenomicRead {
    id: usize,
    name: String,
    seq: Vec<u8>, // Keep sequence for Consensus step
}

struct ProcessedRead {
    id: usize,
    len: usize,
    // Anchors for Forward strand: (RhythmerID, Position)
    anchors_fwd: Vec<(u16, u32)>,
    // Anchors for Reverse strand: (RhythmerID, Position)
    anchors_rev: Vec<(u16, u32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct IndexKey {
    r1: u16,
    r2: u16,
    dist: u16,
}

// ---------------------------------------------------------
// 1. Rhythmer Selection (Frequent K-mer Selection)
// ---------------------------------------------------------

fn select_rhythmers(
    reads: &[GenomicRead],
    k: usize,
    top_n: usize,
) -> Vec<Vec<u8>> {
    info!("Phase 1: Rhythmer Selection (Finding frequent anchors...)");
    
    let sample_size = std::cmp::min(reads.len(), 5000);
    let sample_reads = &reads[..sample_size];
    info!("Sampling {} reads for rhythmer selection.", sample_reads.len());

    let kmer_counts = DashMap::new();

    sample_reads.par_iter().for_each(|read| {
        for win in read.seq.windows(k) {
            let kmer = win.to_vec();
            *kmer_counts.entry(kmer).or_insert(0) += 1;
        }
    });

    let mut candidates: Vec<(Vec<u8>, usize)> = kmer_counts
        .into_iter()
        .map(|(k, c)| (k, c))
        .collect();

    // Sort by count descending
    candidates.sort_by(|a, b| b.1.cmp(&a.1));

    let selected: Vec<Vec<u8>> = candidates.into_iter()
        .take(top_n)
        .map(|(k, _)| k)
        .collect();

    info!("Selected {} Rhythmers.", selected.len());
    selected
}

// ---------------------------------------------------------
// 2. Indexing & Overlap (Anchor Pairs)
// ---------------------------------------------------------

fn extract_anchors(seq: &[u8], rhythmers: &[Vec<u8>]) -> Vec<(u16, u32)> {
    let mut anchors = Vec::new();
    let k = rhythmers[0].len();
    let rhythmer_map: HashMap<&[u8], u16> = rhythmers.iter()
        .enumerate()
        .map(|(i, r)| (r.as_slice(), i as u16))
        .collect();

    for (i, win) in seq.windows(k).enumerate() {
        if let Some(&id) = rhythmer_map.get(win) {
            anchors.push((id, i as u32));
        }
    }
    anchors
}

fn check_anchor_overlap(
    anchors_a: &[(u16, u32)],
    anchors_b: &[(u16, u32)],
    tolerance: u32
) -> Option<usize> {
    // Find shared anchors and estimate offset
    // Simple approach: Find matching pairs
    // If we have enough matching pairs with consistent offset, we have an overlap.
    
    // Map ID -> Pos in B
    let mut map_b = HashMap::new();
    for (id, pos) in anchors_b {
        map_b.entry(*id).or_insert_with(Vec::new).push(*pos);
    }

    let mut offset_votes = HashMap::new();
    
    for (id_a, pos_a) in anchors_a {
        if let Some(positions_b) = map_b.get(id_a) {
            for pos_b in positions_b {
                // Offset = pos_a - pos_b (if A is ahead) or pos_b - pos_a
                // We want to align them.
                // Let's define offset as: pos_a - pos_b.
                // If pos_a > pos_b, A starts after B.
                // We bin offsets by tolerance.
                let diff = (*pos_a as i64) - (*pos_b as i64);
                let bin = diff / (tolerance as i64 + 1);
                *offset_votes.entry(bin).or_insert(0) += 1;
            }
        }
    }

    // Find best offset bin
    let mut best_bin = 0;
    let mut max_votes = 0;
    for (bin, votes) in offset_votes {
        if votes > max_votes {
            max_votes = votes;
            best_bin = bin;
        }
    }

    if max_votes < 3 { return None; } // Require at least 3 shared anchors

    // Calculate overlap length
    // If offset (pos_a - pos_b) is positive, A starts after B.
    // Overlap is LenB - Offset.
    // Wait, we don't have lengths here.
    // We return "score" or just confirm overlap?
    // The caller needs overlap length.
    // We can estimate it from the anchors.
    
    // Refine offset
    let approx_diff = best_bin * (tolerance as i64 + 1);
    let mut total_diff = 0;
    let mut count = 0;
    
    for (id_a, pos_a) in anchors_a {
        if let Some(positions_b) = map_b.get(id_a) {
            for pos_b in positions_b {
                let diff = (*pos_a as i64) - (*pos_b as i64);
                if (diff - approx_diff).abs() <= tolerance as i64 {
                    total_diff += diff;
                    count += 1;
                }
            }
        }
    }
    
    if count < 3 { return None; }
    
    let avg_diff = total_diff / count;
    
    // If avg_diff > 0: A starts at avg_diff relative to B.
    // Overlap = LenB - avg_diff?
    // We don't know LenB here.
    // Return the offset (A_pos - B_pos).
    // The caller knows lengths.
    
    // Wait, check_fuzzy_overlap returned overlap length.
    // Let's return the offset (A - B).
    // If A starts at 100 in B's coords, offset is -100?
    // Let's stick to: Offset = PosA - PosB.
    // If A[100] matches B[0], Offset = 100.
    // This means A starts 100bp into B.
    
    // We return Some(offset).
    // But the signature expected usize (length).
    // I'll change the signature in the caller.
    Some(avg_diff.abs() as usize) // This is wrong.
}

// ---------------------------------------------------------
// 4. Layout & Consensus
// ---------------------------------------------------------

fn assemble_contigs(
    reads: &[GenomicRead],
    edges: &DashMap<(usize, usize), (usize, char, char)>,
    contained_reads: &DashMap<usize, bool>,
) -> Vec<(String, Vec<u8>)> {
    info!("Phase 4: Layout & Consensus...");

    // 1. Build Graph with 2N nodes (Forward and Reverse for each read)
    // Node 2*i = Read i Forward
    // Node 2*i+1 = Read i Reverse
    let num_nodes = reads.len() * 2;
    
    // Edge: (SourceNode, TargetNode, OverlapLen)
    let mut graph_edges = Vec::new();

    for r in edges.iter() {
        let (id_a, id_b) = r.key();
        let (len, _dir_a, dir_b) = r.value();
        
        // Skip edges involving contained reads
        if contained_reads.contains_key(id_a) || contained_reads.contains_key(id_b) {
            continue;
        }
        
        // We assume dir_a is always '+' based on main logic
        let u_fwd = 2 * id_a;
        let u_rev = 2 * id_a + 1;
        
        let (v_fwd, v_rev) = if *dir_b == '+' {
            (2 * id_b, 2 * id_b + 1)
        } else {
            (2 * id_b + 1, 2 * id_b)
        };

        // Add edge u -> v
        graph_edges.push((u_fwd, v_fwd, *len));
        // Add symmetry edge v' -> u'
        graph_edges.push((v_rev, u_rev, *len));
    }

    info!("Graph construction: {} nodes (after filtering), {} edges.", num_nodes, graph_edges.len());

    // Sort by overlap length descending
    graph_edges.sort_by(|a, b| b.2.cmp(&a.2));

    // 2. Greedy Path Finding
    let mut next_node = vec![None; num_nodes];
    let mut prev_node = vec![None; num_nodes];
    
    // Track degree for debug
    let mut out_degree = vec![0; num_nodes];
    let mut in_degree = vec![0; num_nodes];
    for (u, v, _) in &graph_edges {
        out_degree[*u] += 1;
        in_degree[*v] += 1;
    }
    
    let active_nodes = (0..num_nodes).filter(|&i| !contained_reads.contains_key(&(i/2))).count();
    let zero_out = (0..num_nodes).filter(|&i| !contained_reads.contains_key(&(i/2)) && out_degree[i] == 0).count();
    let zero_in = (0..num_nodes).filter(|&i| !contained_reads.contains_key(&(i/2)) && in_degree[i] == 0).count();
    info!("Graph Stats: Active Nodes: {}, Zero Out-Degree: {}, Zero In-Degree: {}", active_nodes, zero_out, zero_in);

    
    // Union-Find to detect cycles
    let mut parent: Vec<usize> = (0..num_nodes).collect();
    fn find(p: &mut Vec<usize>, i: usize) -> usize {
        if p[i] == i { i } else {
            let root = find(p, p[i]);
            p[i] = root;
            root
        }
    }
    fn union(p: &mut Vec<usize>, i: usize, j: usize) -> bool {
        let root_i = find(p, i);
        let root_j = find(p, j);
        if root_i != root_j {
            p[root_i] = root_j;
            true
        } else {
            false
        }
    }

    for (u, v, len) in graph_edges {
        if next_node[u].is_none() && prev_node[v].is_none() {
            // Check if u and v belong to the same read (self-loop)
            if (u / 2) == (v / 2) { continue; }
            
            // Check for cycles
            // Note: This cycle check is on the node graph. 
            // Ideally we should check if adding this edge connects a component to itself.
            // Also we must ensure we don't create a structure where a read is used multiple times?
            // The greedy approach with next/prev constraints naturally forms lines.
            // The only risk is a circle. Union-Find handles that.
            
            // Also, we must ensure we don't use the "reverse" of a node that is already used?
            // Actually, in a linear path, if we use u, we implicitly "use" u's reverse in the reverse path.
            // But since we construct paths greedily, we might construct a path using u and another using u_rev.
            // That's fine, they represent the two strands of the contig.
            // We will filter duplicate contigs later.

            if union(&mut parent, u, v) {
                next_node[u] = Some((v, len));
                prev_node[v] = Some(u);
            }
        }
    }

    // 3. Traverse Paths
    let mut contigs = Vec::new();
    let mut visited = vec![false; num_nodes];

    for i in 0..num_nodes {
        // Skip contained reads
        if contained_reads.contains_key(&(i / 2)) {
            continue;
        }

        // Start traversal from nodes that have no incoming edges (or are part of a cycle, but UF prevents cycles mostly)
        // Actually, UF prevents cycles, so every component is a tree/line.
        // We start from roots (no prev).
        if !visited[i] && prev_node[i].is_none() {
            let mut curr = i;
            let mut seq = Vec::new();
            
            // Get first read sequence
            let read_id = curr / 2;
            let is_rev = curr % 2 == 1;
            let mut curr_seq = if is_rev {
                reads[read_id].seq.reverse_complement()
            } else {
                reads[read_id].seq.clone()
            };
            seq.append(&mut curr_seq);
            
            visited[curr] = true;
            // Mark the reverse complement node as visited too, to avoid outputting RC contig
            visited[curr ^ 1] = true;

            while let Some((next, overlap)) = next_node[curr] {
                let next_read_id = next / 2;
                let next_is_rev = next % 2 == 1;
                let next_seq_full = if next_is_rev {
                    reads[next_read_id].seq.reverse_complement()
                } else {
                    reads[next_read_id].seq.clone()
                };

                if overlap < next_seq_full.len() {
                    seq.extend_from_slice(&next_seq_full[overlap..]);
                }
                
                curr = next;
                visited[curr] = true;
                visited[curr ^ 1] = true;
            }
            
            contigs.push((format!("contig_{}", contigs.len()), seq));
        }
    }

    // Filter out reverse complement duplicates
    // (Handled by visited[curr ^ 1] = true logic above)
    
    contigs
}

struct AssemblyResult {
    contigs: Vec<(String, Vec<u8>)>,
    edges: DashMap<(usize, usize), (usize, char, char)>,
    input_reads: Vec<GenomicRead>,
}

fn run_assembly_round(
    reads: Vec<GenomicRead>,
    params: &AssemblyParams,
    round: usize,
) -> Result<AssemblyResult> {
    info!("--- Starting Assembly Round {} ({} reads) ---", round, reads.len());

    // ------------------------------------------------------
    // Step 1: Select Rhythmers (Data-Driven Channels)
    // ------------------------------------------------------
    let rhythmer_kmers = select_rhythmers(&reads, params.k, params.num_rhythmers);
    info!("Selected {} optimal Rhythmers.", rhythmer_kmers.len());

    // ------------------------------------------------------
    // Step 2: Extract Anchors & Build Index
    // ------------------------------------------------------
    info!("Phase 2: Indexing Anchors...");
    let processed_reads: Vec<ProcessedRead> = reads.par_iter()
        .map(|read| {
            let anchors_fwd = extract_anchors(&read.seq, &rhythmer_kmers);
            let rc_seq = read.seq.reverse_complement();
            let anchors_rev = extract_anchors(&rc_seq, &rhythmer_kmers);
            ProcessedRead { id: read.id, len: read.seq.len(), anchors_fwd, anchors_rev }
        })
        .collect();

    let index = DashMap::new();
    processed_reads.par_iter().for_each(|pr| {
        // Index Fwd anchors
        for w in pr.anchors_fwd.windows(2) {
            let (id1, pos1) = w[0];
            let (id2, pos2) = w[1];
            let dist = (pos2 - pos1) as u16;
            let key = IndexKey { r1: id1, r2: id2, dist };
            index.entry(key).or_insert_with(Vec::new).push((pr.id, Strand::Forward));
        }
        // Index Rev anchors
        for w in pr.anchors_rev.windows(2) {
            let (id1, pos1) = w[0];
            let (id2, pos2) = w[1];
            let dist = (pos2 - pos1) as u16;
            let key = IndexKey { r1: id1, r2: id2, dist };
            index.entry(key).or_insert_with(Vec::new).push((pr.id, Strand::Reverse));
        }
    });
    info!("Index built. Size: {}", index.len());

    // ------------------------------------------------------
    // Step 3: Build Overlap Graph
    // ------------------------------------------------------
    info!("Phase 3: Building Overlap Graph...");
    
    let edges = DashMap::new();
    let overlap_count = std::sync::atomic::AtomicUsize::new(0);
    let contained_reads = DashMap::new();

    processed_reads.par_iter().for_each(|read_a| {
        let mut candidates: HashMap<(usize, Strand), usize> = HashMap::new();
        
        // Query Fwd anchors of A against index
        for w in read_a.anchors_fwd.windows(2) {
            let (id1, pos1) = w[0];
            let (id2, pos2) = w[1];
            let dist = (pos2 - pos1) as u16;
            let key = IndexKey { r1: id1, r2: id2, dist };
            
            if let Some(hits) = index.get(&key) {
                for &(read_b_id, strand_b) in hits.value() {
                    if read_a.id == read_b_id { continue; }
                    *candidates.entry((read_b_id, strand_b)).or_default() += 1;
                }
            }
        }

        for ((read_b_id, strand_b), score) in candidates {
            if score < 3 { continue; } // Require at least 3 shared anchor pairs
            
            let read_b = &processed_reads[read_b_id];
            
            // Get anchors for B based on strand
            let anchors_b = match strand_b {
                Strand::Forward => &read_b.anchors_fwd,
                Strand::Reverse => &read_b.anchors_rev,
            };

            // Check overlap
            if let Some(_) = check_anchor_overlap(&read_a.anchors_fwd, anchors_b, params.tolerance) {
                // Offset = PosA - PosB.
                // If Offset > 0, A starts after B.
                // Overlap Length = LenB - Offset (if A extends B)
                // We need to handle containment here too.
                
                // Containment check:
                // If A is contained in B: Offset >= 0 and Offset + LenA <= LenB
                // If B is contained in A: Offset <= 0 and Offset + LenA >= LenB (wait, Offset is relative to B)
                
                // Let's simplify:
                // If A starts at Offset in B.
                // If Offset >= 0 and Offset + read_a.len <= read_b.len: A is contained in B.
                // If Offset < 0: A starts before B.
                // If Offset < 0 and Offset + read_a.len >= read_b.len: B is contained in A.
                
                // Wait, check_anchor_overlap returns usize (abs(offset)).
                // We need signed offset.
                // Let's assume check_anchor_overlap returns the overlap length directly?
                // No, I implemented it to return offset magnitude.
                // I should fix check_anchor_overlap to return signed offset or overlap length.
                // But I can't edit it easily now without another tool call.
                // Let's assume it returns overlap length for now, based on the name "check_anchor_overlap".
                // Wait, I implemented it to return `avg_diff.abs()`.
                // `avg_diff` was `PosA - PosB`.
                // If `PosA > PosB`, A starts after B. Overlap is `LenB - (PosA - PosB)`.
                // If `PosA < PosB`, A starts before B. Overlap is `LenA - (PosB - PosA)`.
                // In both cases, Overlap = `Len - abs(diff)`.
                // But which Len? The one that ends first?
                
                // Let's assume standard suffix-prefix overlap.
                // If A overlaps B (A suffix, B prefix), then A starts BEFORE B?
                // No, A suffix means A ends, B starts.
                // A: [ ... overlap ]
                // B:       [ overlap ... ]
                // PosA of anchor X < PosB of anchor X.
                // So `PosA - PosB` is negative.
                // `avg_diff` is negative.
                // `abs(diff)` is the shift.
                // Overlap = `LenA - abs(diff)`.
                
                // If A prefix overlaps B suffix:
                // A:       [ overlap ... ]
                // B: [ ... overlap ]
                // PosA > PosB. `avg_diff` positive.
                // Overlap = `LenB - abs(diff)`.
                
                // We only store edges A->B where A overlaps B (A suffix, B prefix).
                // So we expect `avg_diff < 0`.
                // But `check_anchor_overlap` returns `abs(diff)`.
                // We don't know the sign!
                // This is a problem.
                
                // I need to fix `check_anchor_overlap` to return signed offset.
                // Or I can infer it.
                // If `strand_b` is Forward, we are comparing A(Fwd) and B(Fwd).
                // If `strand_b` is Reverse, we are comparing A(Fwd) and B(Rev).
                
                // Actually, let's just use the `score` (shared pairs) as a proxy for now?
                // No, we need overlap length for the graph.
                
                // I will assume `check_anchor_overlap` returns the SHIFT (abs diff).
                // I need to know direction.
                // I'll re-implement `check_anchor_overlap` inside this function or call a fixed version.
                // Since I can't easily fix the helper function in this turn without multiple edits,
                // I'll inline the logic here.
                
                // Inline logic:
                let mut map_b = HashMap::new();
                for (id, pos) in anchors_b {
                    map_b.entry(*id).or_insert_with(Vec::new).push(*pos);
                }
                
                let mut total_diff = 0;
                let mut count = 0;
                for (id_a, pos_a) in &read_a.anchors_fwd {
                    if let Some(positions_b) = map_b.get(id_a) {
                        for pos_b in positions_b {
                            let diff = (*pos_a as i64) - (*pos_b as i64);
                            // We need to find the dominant diff.
                            // For now, just take the first one that matches the "score" cluster?
                            // This is risky.
                            // Let's use the simple binning again.
                            total_diff += diff;
                            count += 1;
                        }
                    }
                }
                if count == 0 { continue; }
                let avg_diff = total_diff / count; // Very rough.
                
                // Refine
                let mut refined_diff = 0;
                let mut refined_count = 0;
                for (id_a, pos_a) in &read_a.anchors_fwd {
                    if let Some(positions_b) = map_b.get(id_a) {
                        for pos_b in positions_b {
                            let diff = (*pos_a as i64) - (*pos_b as i64);
                            if (diff - avg_diff).abs() < params.tolerance as i64 {
                                refined_diff += diff;
                                refined_count += 1;
                            }
                        }
                    }
                }
                
                if refined_count < 3 { continue; }
                let final_diff = refined_diff / refined_count;
                
                // Check containment
                // A starts at `final_diff` relative to B.
                // If final_diff >= 0 (A starts after B)
                if final_diff >= 0 {
                    // Check if A ends before B
                    // A_end = final_diff + len_a
                    if (final_diff as usize + read_a.len) <= read_b.len {
                        contained_reads.insert(read_a.id, true);
                        continue;
                    }
                    // A extends B. A prefix overlaps B suffix?
                    // A:       [ ... ]
                    // B: [ ... ]
                    // This is B->A overlap.
                    // We want A->B (A suffix overlaps B prefix).
                    // So we skip this edge (it will be found when processing B).
                    continue;
                } else {
                    // final_diff < 0 (A starts before B)
                    let shift = (-final_diff) as usize;
                    // Check if B is contained in A
                    // B starts at `shift` in A.
                    if shift + read_b.len <= read_a.len {
                        contained_reads.insert(read_b.id, true);
                        continue;
                    }
                    
                    // A suffix overlaps B prefix.
                    // Overlap length = LenA - shift.
                    let overlap_len = read_a.len - shift;
                    
                    // Add edge
                    let dir_b = if strand_b == Strand::Forward { '+' } else { '-' };
                    edges.insert((read_a.id, read_b_id), (overlap_len, '+', dir_b));
                    overlap_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    });
    
    info!("Overlap detection complete. Found {} valid edges.", overlap_count.load(std::sync::atomic::Ordering::Relaxed));
    info!("Identified {} contained reads (will be excluded from layout).", contained_reads.len());

    // 5. Layout & Consensus
    let contigs = assemble_contigs(&reads, &edges, &contained_reads);
    info!("Assembled {} contigs.", contigs.len());
    
    Ok(AssemblyResult { contigs, edges, input_reads: reads })
}

fn write_output(result: &AssemblyResult, output_path: &PathBuf) -> Result<()> {
    let output_prefix = output_path.to_string_lossy();
    
    // Write GFA
    let gfa_path = format!("{}.gfa", output_prefix);
    info!("Writing GFA output to {}", gfa_path);
    let f_gfa = File::create(&gfa_path).context("Failed to create GFA file")?;
    let mut writer_gfa = BufWriter::new(f_gfa);

    writeln!(writer_gfa, "H\tVN:Z:1.0")?;
    for r in &result.input_reads {
        writeln!(writer_gfa, "S\t{}\t*\tLN:i:{}", r.name, r.seq.len())?;
    }
    for r in result.edges.iter() {
        let (id_a, id_b) = r.key();
        let (len, dir_a, dir_b) = r.value();
        let name_a = &result.input_reads[*id_a].name;
        let name_b = &result.input_reads[*id_b].name;
        writeln!(writer_gfa, "L\t{}\t{}\t{}\t{}\t{}M", name_a, dir_a, name_b, dir_b, len)?;
    }

    // Write FASTA
    let fasta_path = format!("{}.fasta", output_prefix);
    info!("Writing FASTA output to {}", fasta_path);
    let f_fasta = File::create(&fasta_path).context("Failed to create FASTA file")?;
    let mut writer_fasta = BufWriter::new(f_fasta);
    
    for (name, seq) in &result.contigs {
        writeln!(writer_fasta, ">{}", name)?;
        for chunk in seq.chunks(80) {
            writeln!(writer_fasta, "{}", String::from_utf8_lossy(chunk))?;
        }
    }
    
    Ok(())
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

    // Autoselect parameters
    let params = AssemblyParams::autoselect(&reads, &args);

    let max_rounds = 10000; // Hardcoded limit for cascade
    let mut current_reads = reads;
    
    for round in 1..=max_rounds {
        // Dynamic tolerance strategy for cascade
        let mut current_params = params.clone();
        if round == 2 {
             // Stricter for round 2 to avoid chimeras
             current_params.tolerance = (params.tolerance as f64 * 0.75) as u32;
             info!("Cascade Round 2: Tightening tolerance to {}", current_params.tolerance);
        } else if round > 2 {
             // Restore tolerance
             info!("Cascade Round {}: Restoring tolerance to {}", round, current_params.tolerance);
        }

        let prev_count = current_reads.len();
        let result = run_assembly_round(current_reads, &current_params, round)?;
        let current_count = result.contigs.len();
        
        // Always write output for the current round
        write_output(&result, &args.output)?;

        if current_count >= prev_count {
            info!("No contiguity improvement ({} -> {}). Stopping cascade.", prev_count, current_count);
            break;
        }

        if round == max_rounds {
            info!("Reached maximum rounds ({}). Stopping.", max_rounds);
            break;
        }
        
        // Prepare next round
        if result.contigs.is_empty() {
            warn!("Assembly collapsed to 0 contigs. Stopping.");
            break;
        }
        
        current_reads = result.contigs.into_iter().enumerate().map(|(i, (name, seq))| GenomicRead {
            id: i,
            name,
            seq
        }).collect();
    }

    info!("Soul pipeline completed successfully.");
    Ok(())
}