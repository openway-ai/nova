#!/bin/bash
# 测试修复后的chat脚本

echo "Testing fixed chat script with a simple prompt..."
echo ""

# 使用简单的测试prompt
echo "What is 2+2?" | bash /mnt/shared-storage-user/puyuan/code/nova/nova/ebt/runs/chat_ebt.sh --max-tokens 50

echo ""
echo "Test completed. Check if output is coherent (not gibberish)."
