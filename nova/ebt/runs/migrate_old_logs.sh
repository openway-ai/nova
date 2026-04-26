#!/bin/bash
################################################################################
# 一次性迁移脚本: 将旧的散乱日志归档到统一目录
# 运行后可删除本脚本
################################################################################

set -e

LOGS_DIR="/mnt/shared-storage-user/puyuan/code/nova/nova/ebt/logs"
ARCHIVE_DIR="${LOGS_DIR}/logs_before20260324"

mkdir -p "${ARCHIVE_DIR}/summary"
mkdir -p "${ARCHIVE_DIR}/inference_logs"

echo "📦 归档目录: ${ARCHIVE_DIR}"
echo ""

# 1. 移动 logs/ 根目录下的 .log 文件 (eval_nanochat_shards_*.log 等)
COUNT=0
for f in "${LOGS_DIR}"/*.log; do
    [ -f "$f" ] || continue
    mv "$f" "${ARCHIVE_DIR}/"
    COUNT=$((COUNT + 1))
done
echo "  ✅ 根目录 .log 文件: ${COUNT} 个"

# 2. 移动 eval/summary/ 下的 .txt 文件
COUNT=0
for f in "${LOGS_DIR}"/eval/summary/*.txt; do
    [ -f "$f" ] || continue
    mv "$f" "${ARCHIVE_DIR}/summary/"
    COUNT=$((COUNT + 1))
done
echo "  ✅ summary .txt 文件: ${COUNT} 个"

# 3. 移动 eval/inference/ 下散落的 .log 文件
COUNT=0
while IFS= read -r f; do
    mv "$f" "${ARCHIVE_DIR}/inference_logs/"
    COUNT=$((COUNT + 1))
done < <(find "${LOGS_DIR}/eval/inference" -name "*.log" -type f 2>/dev/null)
echo "  ✅ inference .log 文件: ${COUNT} 个"

# 4. 清理空的 summary 目录
rmdir "${LOGS_DIR}/eval/summary" 2>/dev/null && echo "  🧹 删除空目录: eval/summary/" || true

echo ""
echo "📁 归档结果:"
echo "  $(find "${ARCHIVE_DIR}" -type f | wc -l) 个文件已归档到:"
echo "  ${ARCHIVE_DIR}/"
echo ""
echo "🗑️  本脚本可安全删除: rm $(realpath "$0")"
