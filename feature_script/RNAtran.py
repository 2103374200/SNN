def convert_rna_to_complement(input_file, output_file):
    # 定义RNA互补碱基对
    complement = {"A": "U", "U": "A", "C": "G", "G": "C"}

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            # 如果是描述行，直接写入输出文件
            if line.startswith(">") or line.startswith("#"):
                outfile.write(line)
            else:
                # 转换序列为互补序列，跳过无效字符
                complement_seq = "".join(complement[base] for base in line.strip() if base in complement)
                reverse_complement_seq = complement_seq[::-1]  # 反转序列
                outfile.write(reverse_complement_seq + "\n")

# 调用函数
input_file = "../data/trainset.txt"  # 替换为你的输入文件名
output_file = "../data/traintran.txt"  # 输出文件名
convert_rna_to_complement(input_file, output_file)
print("转换完成，结果已保存到", output_file)
