tail -f /workspace/launch.log
grep -i "error\|exception\|traceback\|cuda" /workspace/launch.log | head -50
cat /workspace/output/model/training.log
