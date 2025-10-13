# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:22:57 2018
Modified to handle multiple input files and merge outputs directly

@author: khanhle
"""
import re
import sys
exec(open('/home/featurize/work/evosnr/preprocessing/tools/configurator_preprocessing.py').read()) # overrides from command line or config file



target_species=target
# Define input and output file paths
in_files = [
   
    target_fna
]
out_file = f'{cache_path}/merged_output.txt'


def sequence_to_kmers(sequence, k=8):
    """
    Split a sequence into overlapping k-mers, joined by spaces.
    """
    if len(sequence) < k:
        return ""  # Skip sequences shorter than k
    kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
    return " ".join(kmers)

def process_and_merge_files(input_files, output_file, k=6):
    """
    Process multiple FASTA files, split sequences into k-mers, and merge into a single output file
    in FastText format (space-separated k-mers).
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as fout:
            for in_file in input_files:
                print(f"Processing file: {in_file}")
                spe_character = []
                
                with open(in_file, 'r', encoding='utf-8') as f:
                    sequence = ""
                    for line in f:
                        line = line.strip()
                        # Skip header lines
                        if line.startswith('>'):
                            # Process the previous sequence if it exists
                            if sequence:
                                kmers_line = sequence_to_kmers(sequence, k)
                                if kmers_line:  # Only write non-empty k-mer lines
                                    fout.write(kmers_line + '\n')
                            sequence = ""  # Reset for the next sequence
                        else:
                            # Accumulate sequence, remove special characters if needed
                            if not any(x in line for x in spe_character):
                                sequence += line.replace('\n', '').upper()  # Convert to uppercase
                    # Process the last sequence
                    if sequence:
                        kmers_line = sequence_to_kmers(sequence, k)
                        if kmers_line:
                            fout.write(kmers_line + '\n')
        
        print(f"Files processed and merged successfully into {output_file}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Execute the processing and merging
process_and_merge_files(in_files, out_file)




#this is step2
import fasttext

model = fasttext.train_unsupervised(
    input=out_file,
    model="skipgram",
    lr=0.1,
    dim=100,
    ws=5,
    epoch=100,
    minn=2,  # Minimum n-gram length for k-mers
    maxn=6,  # Maximum n-gram length for k-mers
    wordNgrams=1  # Disable word-level n-grams since k-mers are our "words"
)
model.save_model(fasttext_model_output)