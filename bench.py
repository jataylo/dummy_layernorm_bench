import torch
import time

def layer_norm_test(fs=4096, bs=4096, dtype=torch.float):
    ln = torch.nn.LayerNorm((fs,), device="cuda", dtype=dtype)
    X = torch.randn(bs, fs, device="cuda", dtype=dtype, requires_grad=True)
    gO = torch.rand_like(X)

    # Forward
    ln(X)

    # Backward
    X.grad = None
    ln.zero_grad(set_to_none=True)
    out = ln(X)
    out.backward(gO)

    # Synchronize before returning
    torch.cuda.synchronize()

    return

loops = 10

for dtype in [torch.float, torch.half]:
    start_time = time.time()
    for _ in range(loops):
        layer_norm_test(dtype=dtype)
    end_time = time.time()
    elapsed_time = end_time - start_time
    loops_per_sec = loops / elapsed_time
    print(f"Loops per second {dtype}: {loops_per_sec}")

