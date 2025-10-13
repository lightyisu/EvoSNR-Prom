import csv
import subprocess
from itertools import product
import subprocess
import re
exec(open('/root/autodl-tmp/zwk/evosnr_0605/preprocessing/tools/configurator_preprocessing.py').read()) # overrides from command line or config file
import itertools
import re
import itertools

# Define degenerate codes and their corresponding nucleotides
degenerate_codes = {
    'R': ['A', 'G'],
    'Y': ['C', 'T'],
    'M': ['A', 'C'],
    'K': ['G', 'T'],
    'S': ['C', 'G'],
    'W': ['A', 'T'],
    'H': ['A', 'C', 'T'],
    'B': ['C', 'G', 'T'],
    'V': ['A', 'C', 'G'],
    'D': ['A', 'G', 'T'],
    'N': ['A', 'C', 'G', 'T']
}

def expand_sequence(consensus):
    """Expand a consensus sequence with degenerate codes into all possible DNA sequences."""
    # Find positions with degenerate codes
    degen_positions = [i for i, char in enumerate(consensus) if char in degenerate_codes]
    if not degen_positions:
        return [consensus]
    else:
        # Get the possibilities for each degenerate position
        possibilities = [degenerate_codes[consensus[i]] for i in degen_positions]
        # Generate all combinations
        all_combinations = list(itertools.product(*possibilities))
        # For each combination, build the sequence
        sequences = []
        for combo in all_combinations:
            seq_list = list(consensus)
            for pos, nucleotide in zip(degen_positions, combo):
                seq_list[pos] = nucleotide
            sequences.append(''.join(seq_list))
        return sequences

# Read the input file
with open(streme_path, 'r') as f:
    lines = f.readlines()

# Regular expression pattern to match motif lines
motif_pattern = re.compile(r"MOTIF \d+-(\S+) STREME-\d+")

# Extract consensus sequences
motifs = []
for line in lines:
    match = motif_pattern.search(line)
    if match:
        consensus = match.group(1).upper()  # Ensure uppercase
        motifs.append(consensus)

# Write all possible sequences to the output file
with open(lexicon_path, 'w') as outf:
    for consensus in motifs:
        sequences = expand_sequence(consensus)
        for seq in sequences:
            outf.write(seq + '\n')

# Optional: Verify that 40 motifs were processed
print(f"Processed {len(motifs)} motifs.")