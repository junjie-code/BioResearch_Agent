# tools/sequence_analysis.py
"""
DNA/蛋白质序列分析工具

使用 Biopython 进行基础序列分析，包括：
- GC 含量计算
- 序列长度和碱基组成
- 反向互补序列
- 简单的序列特征识别
"""
from langchain_core.tools import tool
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction


@tool
def sequence_analysis(sequence: str, analysis_type: str = "full") -> str:
    """
    分析 DNA 或蛋白质序列的基本特征。

    当用户提供一段生物序列并希望了解其基本性质时使用此工具。

    Args:
        sequence: DNA 序列（如 "ATCGATCG"）或蛋白质序列（如 "MVLSPADKTNVKAAWGKVG"）
        analysis_type: 分析类型
            - "full": 完整分析（默认）
            - "gc": 仅 GC 含量
            - "complement": 反向互补序列
            - "composition": 碱基/氨基酸组成

    Returns:
        格式化的序列分析结果
    """
    # 清理输入序列
    sequence = sequence.strip().upper().replace(" ", "").replace("\n", "")

    if not sequence:
        return "错误：请提供有效的序列。"

    # 判断序列类型
    dna_bases = set("ATCGN")
    is_dna = all(base in dna_bases for base in sequence)

    seq_obj = Seq(sequence)
    results = []

    if is_dna:
        results.append(f"序列类型: DNA")
        results.append(f"序列长度: {len(sequence)} bp")

        if analysis_type in ("full", "gc"):
            gc = gc_fraction(seq_obj) * 100
            results.append(f"GC 含量: {gc:.2f}%")

        if analysis_type in ("full", "composition"):
            results.append(f"碱基组成:")
            for base in "ATCG":
                count = sequence.count(base)
                pct = count / len(sequence) * 100
                results.append(f"  {base}: {count} ({pct:.1f}%)")

        if analysis_type in ("full", "complement"):
            rev_comp = str(seq_obj.reverse_complement())
            results.append(f"反向互补: {rev_comp}")

        if analysis_type == "full":
            # 简单 ORF 检测
            if "ATG" in sequence:
                atg_pos = sequence.index("ATG")
                results.append(f"首个起始密码子(ATG)位置: {atg_pos + 1}")
            else:
                results.append("未发现起始密码子(ATG)")

            # 翻译（如果长度是3的倍数且包含ATG）
            if len(sequence) >= 3:
                try:
                    protein = str(seq_obj.translate())
                    results.append(f"翻译产物: {protein[:50]}{'...' if len(protein) > 50 else ''}")
                except Exception:
                    results.append("翻译失败（序列可能包含非标准碱基）")
    else:
        # 蛋白质序列分析
        results.append(f"序列类型: 蛋白质（推测）")
        results.append(f"序列长度: {len(sequence)} aa (氨基酸)")

        if analysis_type in ("full", "composition"):
            from collections import Counter
            aa_count = Counter(sequence)
            results.append("氨基酸组成（前10种）:")
            for aa, count in aa_count.most_common(10):
                pct = count / len(sequence) * 100
                results.append(f"  {aa}: {count} ({pct:.1f}%)")

        if analysis_type == "full":
            # 估算分子量（粗略）
            avg_aa_mw = 110  # 平均氨基酸分子量约 110 Da
            est_mw = len(sequence) * avg_aa_mw
            results.append(f"估算分子量: ~{est_mw:,} Da ({est_mw/1000:.1f} kDa)")

    return "\n".join(results)