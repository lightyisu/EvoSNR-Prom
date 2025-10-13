from .lexicon import count_kmers_in_dataset_with_vocab
def countLexiconKmer(datatsetBar,kmer_vocab,total_kmer_counts):
    #训练集频率统计
    for data in datatsetBar:
        X, Y = data
        batch_kmer_counts = count_kmers_in_dataset_with_vocab(X, kmer_vocab)
        
            # 累加当前批次的k-mer频率到总计中
        for kmer, count in batch_kmer_counts.items():
                total_kmer_counts[kmer] += count
    return total_kmer_counts