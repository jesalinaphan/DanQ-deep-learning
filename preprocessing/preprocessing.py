#!/usr/bin/env python3

"""
Generates data.csv
Expects the following files to be present in the data directory:
- mm10.fa
- mm10_200bp_bins.bed
- All peak files in the peaks directory
"""
from pybedtools import BedTool
from pyfaidx import Fasta
import numpy as np
from tqdm import tqdm
import os
import csv

class PreprocessMouseDanQ:
    def __init__(self, genome_fasta, bins_bed, peak_files):
        self.genome = Fasta(genome_fasta)
        self.peak_files = peak_files

        print("[1] Loading 200-bp bins...")
        all_bins = BedTool(bins_bed)

        print("[2] Merging all peaks...")
        all_peaks = BedTool.cat(*[BedTool(p) for p in peak_files], postmerge=False)

        print("[3] Filtering bins that overlap any peak...")
        filtered_bins = all_bins.intersect(all_peaks, u=True)

        self.samples = []

        print("[4] Processing bins and building dataset...")
        for bin in tqdm(filtered_bins[:50]):
            chrom, start, end = bin.chrom, int(bin.start), int(bin.end)
            center = (start + end) // 2
            seq_start = center - 500
            seq_end = center + 500

            if seq_start < 0:
                continue
            try:
                seq = self.genome[chrom][seq_start:seq_end].seq.upper()
            except KeyError:
                continue
            if len(seq) != 1000 or not all(base in "ACGT" for base in seq):
                continue

            # Build binary target vector
            target_vector = np.zeros(len(peak_files), dtype=np.uint8)
            bin_region = BedTool(f"{chrom}\t{start}\t{end}", from_string=True)

            for i, peak_file in enumerate(peak_files):
                if bin_region.intersect(peak_file, u=True):
                    target_vector[i] = 1

            self.samples.append((seq, ''.join(map(str, target_vector))))

    def save_to_csv(self, out_path):
        print(f"[5] Saving {len(self.samples)} samples to {out_path}...")
        with open(out_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["sequence", "target"])  # header
            for seq, target_str in self.samples:
                writer.writerow([seq, target_str])
        print("Done.")

if __name__ == "__main__":
    genome_fasta = "./data/mm10.fa"
    bins_bed = "./data/mm10_200bp_bins.bed"
    peak_dir = "./data/peaks/"
    peak_files = [os.path.join(peak_dir, f) for f in os.listdir(peak_dir) if f.endswith(".bed")]
    out_path = "./data/processed.csv"

    assert os.path.exists(genome_fasta), f"Genome fasta file not found: {genome_fasta}"
    assert os.path.exists(bins_bed), f"Bins bed file not found: {bins_bed}"
    assert os.path.exists(peak_dir), f"Peak directory not found: {peak_dir}"
    assert len(peak_files) > 0, "No peak files found in the specified directory."

    dataset_builder = PreprocessMouseDanQ(genome_fasta, bins_bed, peak_files)
    dataset_builder.save_to_csv(out_path)