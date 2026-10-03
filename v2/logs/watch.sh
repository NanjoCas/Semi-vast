#!/bin/bash
# 实时打印训练进度：bash logs/watch.sh [日志文件]，默认看最新的日志；Ctrl+C 退出，不影响训练。
# 关键行（步骤、每个 epoch 的结果、测试结果）逐行打印，进度条在最后一行原地刷新。
cd /root/autodl-tmp/Semi-vast/v2
LOG=${1:-$(ls -t logs/*.log | head -1)}
echo "日志: $LOG    （Ctrl+C 退出，不影响训练）"
tail -c 300000 -F "$LOG" 2>/dev/null | /root/miniconda3/bin/python -u -c '
import codecs, os, re, shutil, sys
KEEP = re.compile(r"^\[(STEP|DONE|FAIL|SKIP)\]|SUCCESS|epoch_mode=|Epoch \d+/\d+ loss=|New best|\[val/epoch\d+\]|\[test\]"
                  r"|\[Epoch \d+\] train_loss|\[pseudo_label_quality\]|set \w+ +accuracy|full pool accuracy|Traceback|Error")
BAR = re.compile(r"^(train epoch|Epoch \d+/\d+:|val/epoch|test:|Scoring batches|DeBERTa inference)")
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
