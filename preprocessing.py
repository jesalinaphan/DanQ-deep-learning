'''
Loads and preprocesses the data used to train DanQ

DanQ peprocessing steps, accoording to ChatGPT:
1. Segment genome into 200-bp non-overlapping bins
2. Get 1000-bp sequence centered on each 200-bp bin
- For each bin (200 bp), center a 1000-bp window around it.
- So:
    If your bin starts at position x, then:
    center = x + 100
    The 1000-bp window goes from center - 500 to center + 500
3. Generate the binary target vector
- Use public ChIP-seq and DNase-seq peak BED files (from ENCODE and Roadmap).
- Intersect each 200-bp bin with these 919 peak sets (so 919 peak files)
- If a bin overlaps a peak in any dataset → set the corresponding index in the target vector to 1.
- Result: each sample has a 919-dimensional binary target vector.
4. Filter samples
- Keep only bins whose 1000-bp sequence overlaps at least one ChIP-seq or DNase-seq peak.
5. One-hot encode the DNA sequence
- Encode each nucleotide (A, C, G, T) as a 4-dimensional binary vector.
    A = [1, 0, 0, 0]
    C = [0, 1, 0, 0]
    G = [0, 0, 1, 0]
    T = [0, 0, 0, 1]
    Unknown or ambiguous bases (e.g., N) → all zeros [0, 0, 0, 0]

Notes: 
- For us, we will use the mm39 genome assembly for the mouse genome.
- We can use a smaller number of peak files (eg 4 instead of 919) for simplicity.
- We need to peak files corresponding to the mm39 genome assembly
'''

#Example preprocessing code:

NUC_TO_ONEHOT = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1]
}

def one_hot_encode(seq):
    return np.array([NUC_TO_ONEHOT.get(base.upper(), [0, 0, 0, 0]) for base in seq])

class MouseDanQDataset(Dataset):
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

        print("[4] Processing bins...")
        for bin in tqdm(filtered_bins):
            chrom, start, end = bin.chrom, int(bin.start), int(bin.end)
            center = (start + end) // 2
            seq_start = center - 500
            seq_end = center + 500

            if seq_start < 0:
                continue
            try:
                seq = self.genome[chrom][seq_start:seq_end].seq
            except KeyError:
                continue
            if len(seq) != 1000:
                continue

            # Build target vector
            target_vector = np.zeros(len(peak_files), dtype=np.float32)
            bin_region = BedTool(f"{chrom}\t{start}\t{end}", from_string=True)

            for i, peak_file in enumerate(peak_files):
                if bin_region.intersect(peak_file, u=True):
                    target_vector[i] = 1

            self.samples.append((seq, target_vector))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        onehot = one_hot_encode(seq)  # shape: (1000, 4)
        onehot = torch.tensor(onehot).permute(1, 0).float()  # shape: (4, 1000)
        target = torch.tensor(target)
        return onehot, target

#And then to use ...
from build_danq_mouse_dataset import MouseDanQDataset

genome_fasta = "mm39.fa"
bins_bed = "200bp_bins.bed"
peak_files = [
    "peaks/MEL_GATA1.bed",
    "peaks/MEL_NFYA.bed",
    "peaks/CH12_DNase.bed",
    "peaks/CH12_H3K27ac.bed"
]

dataset = MouseDanQDataset(genome_fasta, bins_bed, peak_files)

print("Total samples:", len(dataset))
x, y = dataset[0]
print("Input shape:", x.shape)  # (4, 1000)
print("Target shape:", y.shape)  # (len(peak_files),)

#How to get the 200bp bins?? (Bash)
samtools faidx mm39.fa
cut -f1,2 mm39.fa.fai > mm39.genome
bedtools makewindows -g mm39.genome -w 200 > 200bp_bins.bed
