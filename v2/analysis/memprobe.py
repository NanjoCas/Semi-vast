# detector 显存/速度探针：在 v2/ 下运行 python analysis/memprobe.py <是否开梯度检查点 0|1> <序列长度>。16+48 条序列一次前向+反向。
import sys, time, torch
sys.path.insert(0,".")
from models.detector import DualChannelDetector
ckpt=sys.argv[1]=="1"; L=int(sys.argv[2])
torch.backends.cuda.matmul.allow_tf32=True
d=DualChannelDetector("microsoft/deberta-v3-large",3,cache_dir="../model_cache").cuda().float()
d.train()
if ckpt: d.enable_gradient_checkpointing()
opt=torch.optim.AdamW(d.parameters(),lr=1e-5)
try:
    ts=[]
    for it in range(4):
        torch.cuda.synchronize(); t=time.time()
        with torch.amp.autocast("cuda",dtype=torch.bfloat16):
            a=d.forward_reasoning(torch.randint(5,1000,(16,L)).cuda(),torch.ones(16,L,dtype=torch.long).cuda())
            b=d.forward_reasoning(torch.randint(5,1000,(48,L)).cuda(),torch.ones(48,L,dtype=torch.long).cuda())
            loss=a.float().mean()+b.float().mean()
        loss.backward(); opt.step(); opt.zero_grad(); torch.cuda.synchronize(); ts.append(time.time()-t)
    print(f"checkpointing={ckpt!s:5} L={L}: peak={torch.cuda.max_memory_allocated()/2**30:.1f} GiB  step={min(ts[1:]):.2f}s")
except torch.OutOfMemoryError:
    print(f"checkpointing={ckpt!s:5} L={L}: OOM")
