#!/bin/bash
# Generates the non-overlapping 200bp sequences from the mm10 genome

set -e # Exit immediately if a command exits with a non-zero status

samtools faidx "./data/mm10.fa"
cut -f1,2 "./data/mm10.fa.fai" > "./data/mm10.genome"
bedtools makewindows -g "./data/mm10.genome" -w 200 > "./data/mm10_200bp_bins.bed"
rm "./data/mm10.genome"
rm "./data/mm10.fa.fai"