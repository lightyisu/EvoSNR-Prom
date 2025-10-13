import csv
import subprocess
from itertools import product
import subprocess
import re
exec(open('/root/autodl-tmp/zwk/evosnr_0605/preprocessing/tools/configurator_preprocessing.py').read()) # overrides from command line or config file



#change to pca_file input
def extract_motifs(input_file,output_file):
    
      
        print(f"Starting conversion from {input_file} to FASTA format...")
        # 打开输入文件和输出文件
        with open(input_file, 'r', newline='') as infile, open(output_file, 'w') as outfile:
            reader = csv.reader(infile, delimiter='\t')
            count = 0

            for i, row in enumerate(reader, 1):
                if not any(row) or len(row) < 2:
                    print(f"Warning: Line {i} is malformed. Skipping.")
                    continue

                sequence = row[0].strip().upper()
                labels = row[1].strip()

                if len(sequence) != len(labels):
                    print(f"Warning: Line {i} sequence and label length mismatch. Skipping.")
                    continue

                # 提取对应 label 为 1 的位置
                positive_seq = ''.join([base for base, label in zip(sequence, labels) if label == '1'])

                if not positive_seq:
                    continue  # 如果没有任何正样本，跳过

                count += 1
                identifier = f"seq_{i}"
                outfile.write(f">{identifier}\n{positive_seq}\n")
        print(f"转换完成，结果已保存到 {output_file}")

        print("预处理fa文件处理完毕！")
        # 👇 确保目录存在，不存在就创建
        import os
        os.makedirs(streme_output_dir, exist_ok=True)        # 创建 streme 输出目录
   


        # 定义 MEME 命令
        streme_command = [
            "streme",
            "--p",
            output_file,
            "--dna",
            "--minw",
            "4",
            "--maxw",
            "15",
            "--nmotifs", "60",
            # "--text",
            "--oc", streme_output_dir
            
        ]
        iupac_codes = {
            'A': ['A'], 'C': ['C'], 'G': ['G'], 'T': ['T'],
            'M': ['A', 'C'], 'R': ['A', 'G'], 'W': ['A', 'T'],
            'S': ['C', 'G'], 'Y': ['C', 'T'], 'K': ['G', 'T'],
            'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
            'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'],
            'N': ['A', 'C', 'G', 'T']
        }
        
        # 展开模糊核苷酸为所有可能序列
        def expand_ambiguous_sequence(sequence):
            # 为每个字符生成可能碱基列表
            possible_bases = [iupac_codes.get(c, [c]) for c in sequence]
            # 生成所有组合
            expanded_sequences = [''.join(combo) for combo in product(*possible_bases)]
            return expanded_sequences

        # 用于保存共识序列的列表
        consensus_sequences = []

        # 正则表达式匹配 Motif 名称中的序列（格式如 X-SEQUENCE）
        consensus_pattern = re.compile(r"MOTIF\s+\d+-([A-Z]+)\s*STREME-\d+")

        # 执行命令并实时处理输出
        try:
            with subprocess.Popen(
                streme_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            ) as proc:
                # 实时读取输出
                for line in iter(proc.stdout.readline, ''):
                    print(line, end='')  # 显示进度
                    
                    # 提取 Motif 序列
                    if line.startswith("MOTIF"):
                        match = consensus_pattern.search(line)
                        if match:
                            sequence = match.group(1)  # 提取 GACGATCTCC 等
                            if sequence:
                                consensus_sequences.append(sequence)
                
                proc.wait()
                print(f"\nSTREME执行完成，退出码：{proc.returncode}")
            

        except subprocess.CalledProcessError as e:
            print(f"命令执行失败: {e}")
        except Exception as e:
            print(f"发生错误: {e}")


extract_motifs(target_train_PCA,cache_path+'promoters_meme'+target+'_t.fasta')
