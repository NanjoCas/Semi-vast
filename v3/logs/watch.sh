#!/bin/bash
# 实时打印训练进度：bash logs/watch.sh [日志文件]，默认看 logs/ 下最新的日志；Ctrl+C 退出，不影响训练。
# 关键行（步骤、每次验证的结果、新的最佳点、测试结果、自动检查）逐行打印，进度条在最后一行原地刷新。
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG=${1:-$(ls -t logs/*.log 2>/dev/null | head -1)}
[[ -z "$LOG" ]] && { echo "logs/ 下没有日志"; exit 1; }
echo "日志: $LOG    （Ctrl+C 退出，不影响训练）"
tail -c 300000 -F "$LOG" 2>/dev/null | python -u -c '
import codecs, os, re, shutil, sys
KEEP = re.compile(r"^\[(STEP|DONE|FAIL|SKIP|PLAN|WARN|INFO\] \[|PROTOCOL|SANITY|import)\]?|SUCCESS|budget:|loss: sup|\[step \d+/\d+\]|New best|\[test\]"
                  r"|Training complete|\[Epoch \d+\] train_loss|set \w+ +accuracy|full pool accuracy|PASS|FAIL|结论|Traceback|Error")
BAR = re.compile(r"^(train:|train epoch|Epoch \d+/\d+:|val@|val/|test:|Scoring batches|NLI batches|DeBERTa inference)")
PREFIX = re.compile(r" \[INFO\] \w+ \| ")
dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
buf, bar = "", False

def show_bar(line):
    global bar
    width = shutil.get_terminal_size((120, 20)).columns - 1
    sys.stdout.write("\r\033[K" + line[:width]); sys.stdout.flush(); bar = True

try:
    while True:
        data = os.read(0, 65536)
        if not data:
            break
        parts = re.split(r"[\r\n]", buf + dec.decode(data))
        buf = parts.pop()
        for line in parts:
            line = PREFIX.sub(" ", line.rstrip())
            if KEEP.search(line):
                sys.stdout.write(("\r\033[K" if bar else "") + line + "\n"); bar = False
            elif BAR.search(line):
                show_bar(line)
        if BAR.search(buf):
            show_bar(buf.rstrip())
        sys.stdout.flush()
except KeyboardInterrupt:
    print()
'
