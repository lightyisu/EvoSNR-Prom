#Step1 生成输入序列

from tqdm import tqdm
import os
import csv
from Bio import SeqIO
import pandas as pd
import random

exec(open('/home/featurize/work/evosnr/preprocessing/tools/configurator_preprocessing.py').read()) # overrides from command line or config file

# 指定要读取的记录序号（例如，第0条记录）
RECORD_INDEX = 0  # 可以根据需要更改这个值

#train dev test
file_name=target
fasta_file = target_fna

prom_file =target_prom

output_search_file = f"{cache_path}.res_cache.tsv"

output_file = target_PCA_output


# 读取 FASTA 文件中的基因组序列
def read_fasta(file_path):
    """
    Reads a FASTA file and returns the concatenated sequence (uppercase).
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
        sequence = ''.join(line.strip() for line in lines if not line.startswith(">"))
    return sequence.upper()

# 查找序列中大写字母的索引（TSS 位置）
def find_uppercase_index(seq):
    """
    Finds the index of the first uppercase letter in the sequence (0-based).
    """
    for i, c in enumerate(seq):
        if c.isupper():
            return i
    raise ValueError(f"No uppercase letter found in sequence: {seq}")

# 计算反向互补序列
def reverse_complement(seq):
    """
    Returns the reverse complement of a DNA sequence.
    """
    complement = {'a': 't', 't': 'a', 'c': 'g', 'g': 'c',
                  'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement.get(base, base) for base in reversed(seq))



# 全局搜索启动子序列
def search_promoter_global(genome, pmSequence, strand):
    """
    Performs global search for promoter sequence in the genome.
    Returns list of tuples with (start, end, tss_pos, matched_seq) for all matches.
    """
    len_seq = len(pmSequence)
    matches = []
    
    if strand == "forward":
        search_seq = pmSequence.lower()
    else:
        search_seq = reverse_complement(pmSequence).lower()

    for i in range(0, len(genome) - len_seq + 1):
        extracted = genome[i:i+len_seq].lower()
        if extracted == search_seq:
            if strand == "forward":
                tss_pos = i + find_uppercase_index(pmSequence) + 1  # 1-based
            else:
                tss_pos = i + len_seq - find_uppercase_index(pmSequence)  # 1-based
            matches.append((i+1, i+len_seq, tss_pos, extracted))
    
    return matches



# 主程序：提取启动子位置并进行全局搜索
def extract_promoter_positions(fasta_file, tsv_file, output_file="promoter_search_results.tsv", type_filter=None):
    """
    Extracts promoter positions from the genome based on TSV annotations.
    Performs global search, counts matches, and writes results to file.
    fasta_file: path to the single FASTA file.
    tsv_file: path to the TSV file with promoter annotations.
    output_file: path to the output file.
    type_filter: if specified, only process records with this Type (e.g., 'circular-Chr').
    """
    # 读取基因组序列
    genome = read_fasta(fasta_file)
    genome_length = len(genome)
    print(f"Genome length: {genome_length} bp")

    # 计算 TSV 文件总行数用于进度条（仅统计符合 type_filter 的行）
    with open(tsv_file, "r") as f:
        total_lines = 0
        for line in f:
            if line.strip().startswith("#"):
                continue
            if type_filter:
                columns = line.strip().split("\t")
                if len(columns) >= 2 and columns[1] == type_filter:
                    total_lines += 1
            else:
                total_lines += 1
        total_lines -= 1  # 减去 header

    # 打开输出文件
    with open(output_file, "w") as out_f:
        # 写入表头
        out_f.write("pmId\tType\tStrand\tMatchType\tStart\tEnd\tTSS\tExtractedSeq\tFunctionalSeq\tMatchCount\n")
        
        # 处理 TSV 文件并显示进度条
        with open(tsv_file, "r") as f:

            reader=csv.reader(f)
            header_skipped = False
            total_matches = 0
            
            for line_number, columns in enumerate(tqdm(reader, total=total_lines, desc="Processing promoters")):
                # 跳过注释行
                if columns and columns[0].startswith("#"):
                    continue
                
                if not header_skipped:
                    print(f"Skipping header: {line.strip()}")
                    header_skipped = True
                    continue
                
                
                
                if len(columns) < 12:
                    print(f"Warning: Line {line_number} has fewer than 12 columns: {line.strip()}. Skipping.")
                    continue
              
                # 提取字段
                pmId = columns[0]
                type_name = columns[1]  # Type 列
                strand = columns[9]     # Strand 列（+/-）
                try:
                    posTSS = int(columns[8])  # TSSPosition 列
                except ValueError:
                    print(f"Warning: Invalid posTSS value at line {line_number}: {columns[8]}. Skipping.")
                    continue
                pmSequence = columns[11]  # PromoterSeq 列
                if not pmSequence:
                    print(f"Warning: Empty promoter sequence at line {line_number} (pmId: {pmId}). Skipping.")
                    continue
                # 如果指定了 type_filter，跳过不符合的记录
                if type_filter and type_name != type_filter:
                    continue

                # 查找 TSS 在启动子序列中的索引
                K = find_uppercase_index(pmSequence)
                len_seq = len(pmSequence)

                # 计算初始基因组位置（1-based）
                if strand == "+":  # 正向链
                    start_position = posTSS - K
                    end_position = start_position + len_seq - 1
                else:  # 反向链
                    start_position = posTSS - (len_seq - 1 - K)
                    end_position = posTSS + K

                # 检查初始位置并写入文件
                initial_match = False
                match_count = 0
                if start_position < 1 or end_position > genome_length:
                    print(f"Warning: Initial positions out of range for {pmId} (Type: {type_name}): "
                          f"start={start_position}, end={end_position}")
                else:
                    extracted_seq = genome[start_position-1:end_position].lower()
                    pmSequence_lower = pmSequence.lower()
                    
                    if extracted_seq == pmSequence_lower:
                        initial_match = True
                        functional_seq = reverse_complement(extracted_seq) if strand == "-" else extracted_seq
                        out_f.write(f"{pmId}\t{type_name}\t{strand}\tInitial\t{start_position}\t{end_position}\t"
                                  f"{posTSS}\t{extracted_seq}\t{functional_seq}\t-\n")
                        print(f"{pmId}\t{type_name}\t{strand}\tInitial match at {start_position}-{end_position}")

                if not initial_match:
                    # 进行全局搜索
                    matches = search_promoter_global(genome, pmSequence, "forward" if strand == "+" else "reverse")
                    match_count = len(matches)
                    total_matches += match_count
                    
                    # 写入全局搜索结果
                    if matches:
                        for start, end, tss_pos, matched_seq in matches:
                            functional_seq = reverse_complement(matched_seq) if strand == "-" else matched_seq
                            out_f.write(f"{pmId}\t{type_name}\t{strand}\tGlobal\t{start}\t{end}\t{tss_pos}\t"
                                      f"{matched_seq}\t{functional_seq}\t{match_count}\n")
                
                # 打印每条启动子的摘要
                print(f"{pmId} (Type: {type_name}): Initial match: {initial_match}, Global matches: {match_count}")

            print(f"\nSummary: Found a total of {total_matches} matches across all promoters")
           

    print(f"Results written to {output_file}")






if __name__ == "__main__":
   

    



    
    if not os.path.exists(output_search_file):
     extract_promoter_positions(fasta_file, prom_file, output_search_file, type_filter)
    
    genome_records = list(SeqIO.parse(fasta_file, 'fasta'))  # 解析所有记录为列表

    # 检查记录数量并选择指定记录
    if len(genome_records) == 0:
        raise ValueError("FASTA文件不包含任何记录。")
    elif RECORD_INDEX >= len(genome_records):
        raise ValueError(f"记录序号 {RECORD_INDEX} 超出记录总数 ({len(genome_records)})。")
    else:
        genome_record = genome_records[RECORD_INDEX]  # 选择指定记录
        genome_seq = str(genome_record.seq)
        print(f"正在处理记录 {RECORD_INDEX}: {genome_record.id}")
    

        # 读取search result-TSV文件
        df = pd.read_csv(output_search_file, sep='\t')

        # 参数设置
        TARGET_LENGTH = 180  # 目标序列长度
        MIN_CONTEXT = 20     # 最小上下文长度

        # 用于存储结果的列表
        results = []

        # 遍历TSV文件的每一行
        for index, row in df.iterrows():
            print(row)
            start, end = row['Start'], row['End']  # 原始启动子的起止位置
            start,end=int(start),int(end)
            L = end - start + 1                    # 启动子长度
            C = TARGET_LENGTH - L                  # 上下文总长度

            # 计算左右上下文长度
            if C < 2 * MIN_CONTEXT:
                left = C // 2
                right = C - left
            else:
                left = random.randint(MIN_CONTEXT, C - MIN_CONTEXT)
                right = C - left

            # 计算180bp序列的起止位置
            new_start = max(0, start - 1 - left)          # 序列起始位置（考虑基因组边界）
            new_end = min(len(genome_seq), end + right)   # 序列结束位置（考虑基因组边界）
            seq = genome_seq[new_start:new_end]           # 提取序列

            # 如果序列长度不足180bp，进行调整
            if len(seq) < TARGET_LENGTH:
                if new_start == 0:  # 左侧已到基因组开头，只能向右扩展
                    new_end = min(len(genome_seq), new_end + (TARGET_LENGTH - len(seq)))
                else:  # 左侧有空间，向左扩展
                    new_start = max(0, new_start - (TARGET_LENGTH - len(seq)))
                seq = genome_seq[new_start:new_end]  # 重新提取序列

            # 计算启动子在180bp序列中的位置
            promoter_start_in_seq = (start - 1) - new_start  # 启动子起始索引
            promoter_end_in_seq = end - new_start            # 启动子结束索引

            # 处理边界情况
            if promoter_start_in_seq < 0:          # 左侧超出序列范围
                promoter_start_in_seq = 0
            if promoter_end_in_seq > len(seq):     # 右侧超出序列范围
                promoter_end_in_seq = len(seq)

            # 生成注释序列
            annotation = [0] * len(seq)  # 初始化全0列表
            for i in range(promoter_start_in_seq, promoter_end_in_seq):
                if 0 <= i < len(seq):    # 确保索引有效
                    annotation[i] = 1    # 启动子区域标记为1

            # 将注释列表转换为字符串
            annotation_str = ''.join(map(str, annotation))

            # 将结果添加到列表中
            results.append({
                
                'Sequence': seq,
                'Annotation': annotation_str,
            
            })

        # 将结果转换为DataFrame并保存到TSV文件
        results_df = pd.DataFrame(results)
        
        results_df.to_csv(output_file, sep='\t', index=False,header=False)

        print(f"结果已保存到 '{output_file}'，包含 {len(results)} 条记录。")

    # 删除临时文件 output_search_file
    if os.path.exists(output_search_file):
        os.remove(output_search_file)
        print(f"临时文件 '{output_search_file}' 已删除。")
    else:
        print(f"临时文件 '{output_search_file}' 不存在，无需删除。")