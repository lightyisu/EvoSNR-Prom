import pandas as pd
from collections import Counter
exec(open('Evosnr-main/preprocessing/tools/configurator_preprocessing.py').read()) # overrides from command line or config file

csv_file_list = [source_prom,
                     target_prom]



def get_kmers(sequence, min_k=6, max_k=12):
    sequence = sequence.upper()
    kmers = []
    for k in range(min_k, max_k + 1):
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i + k]
            kmers.append(kmer)
    return kmers

def extract_kmers_by_length(csv_file, min_k=6, max_k=12, top_n_per_length=71):
    df = pd.read_csv(csv_file)
    promoter_sequences = df['PromoterSeq'].tolist()
    kmer_counts_by_length = {k: Counter() for k in range(min_k, max_k + 1)}
    
    for seq in promoter_sequences:
        seq = seq.upper()
        for k in range(min_k, max_k + 1):
            kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
            kmer_counts_by_length[k].update(kmers)
    
    top_kmers = []
    for k in range(min_k, max_k + 1):
        top_kmers.extend(kmer_counts_by_length[k].most_common(top_n_per_length))
    
    return top_kmers

def save_kmers_to_file(kmers, output_file):
    """仅保存k-mer字符串，不包含计数"""
    with open(output_file, 'w') as f:
        for kmer_entry in kmers:
            f.write(f"{kmer_entry[0]}\n")  # 只写入kmer字符串部分，忽略计数

def main():
    
    min_k = 6
    max_k = 10
    top_n = 30
    all_kmers = []
    output_file = kmer_Lexicon_output
    
    for csv_file in csv_file_list:
        top_kmers = extract_kmers_by_length(csv_file, min_k, max_k, top_n)
        all_kmers.extend(top_kmers)
    
    save_kmers_to_file(all_kmers, output_file)
    print('Writer successful')

if __name__ == "__main__":
    main()