import numpy as np
import time

import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

def ulong_to_grid(l):
    l = f"{l:064b}"
    return f"{l[56:64]}\n{l[48:56]}\n{l[40:48]}\n{l[32:40]}\n{l[24:32]}\n{l[16:24]}\n{l[8:16]}\n{l[0:8]}".replace('1', '#').replace('0', '.')

mf = cl.mem_flags
regions_np = np.zeros((20000,), np.ulong)
regions_np[0] = 0
regions_np[1] = (0b00000000
            | (0b00000000 << 8)
            | (0b00000001 << 16)
            | (0b00000001 << 24)
            | (0b00000000 << 32)
            | (0b00000000 << 40)
            | (0b00000000 << 48)
            | (0b00000000 << 56))
regions_np[2] = (0b00000000
            | (0b00000000 << 8)
            | (0b10000000 << 16)
            | (0b10000000 << 24)
            | (0b00000000 << 32)
            | (0b00000000 << 40)
            | (0b00000000 << 48)
            | (0b00000000 << 56))
print(ulong_to_grid(regions_np[1]))
regions_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=regions_np)

neighbords_np = np.zeros((8*len(regions_np)), np.ushort)
neighbords_np[8+0] = 0
neighbords_np[8+1] = 0
neighbords_np[8+2] = 0
neighbords_np[8+3] = 0
neighbords_np[8+4] = 2
neighbords_np[8+5] = 0
neighbords_np[8+6] = 0
neighbords_np[8+7] = 0

neighbors_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=neighbords_np)

prg = cl.Program(ctx, open("main.cl").read()).build()

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, regions_np.nbytes)
sparse_life8 = prg.sparse_life8  # Use this Kernel object for repeated calls
res_np = np.empty_like(regions_np)


start = time.perf_counter()
for i in range(0, 10000):
    sparse_life8(queue, regions_np.shape, None, np.int32(8), regions_g, neighbors_g, res_g)
    sparse_life8(queue, regions_np.shape, None, np.int32(8), regions_g, res_g, neighbors_g)

cl.enqueue_copy(queue, res_np, res_g)

end = time.perf_counter()


print(res_np)
print("time", end - start)
print(ulong_to_grid(res_np[1]))