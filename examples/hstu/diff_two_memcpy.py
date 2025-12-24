import torch
import paged_kvcache_ops
import math
import numpy as np

# tHA_list = [ torch.rand((128, 2, 32, 4, 128), dtype=torch.bfloat16, pin_memory=True) for _ in range(20) ]
# tDA_list = [ torch.rand((128, 2, 32, 4, 128), dtype=torch.bfloat16, device=torch.cuda.current_device()) for _ in range(20) ]

# tHB_list = [ torch.rand((128, 2, 32, 4, 128), dtype=torch.bfloat16, pin_memory=True) for _ in range(20) ]
# tDB_list = [ torch.rand((128, 2, 32, 4, 128), dtype=torch.bfloat16, device=torch.cuda.current_device()) for _ in range(20) ]

# tHA = tHA_list[0]
# tDA = tDA_list[0]
# numelA = tHA.nbytes
# print(numelA, type(numelA))

# tHB = tHB_list[0]
# tDB = tDB_list[0]

# print(tDB.view(-1)[:8])
# print(tHB.view(-1)[:8])

# paged_kvcache_ops.normal_h2d(tDA.data_ptr(), tHA.data_ptr(), tHA.nbytes)
# paged_kvcache_ops.customized_h2d(tDB.data_ptr(), tHB.data_ptr(), tHB.nbytes)

# print(tDB.view(-1)[:8])
# print(tHB.view(-1)[:8])

# print(tDB.view(-1)[-8:])
# print(tHB.view(-1)[-8:])


# torch.cuda.profiler.start()
# for idx in range(10):
#     tHA = tHA_list[idx]
#     tDA = tDA_list[idx]
#     tHB = tHB_list[idx]
#     tDB = tDB_list[idx]
#     paged_kvcache_ops.normal_h2d(tDA.data_ptr(), tHA.data_ptr(), tHA.nbytes)
#     paged_kvcache_ops.customized_h2d(tDB.data_ptr(), tHB.data_ptr(), tHB.nbytes)
#     print()

# for idx in range(10, 20):
#     tHA = tHA_list[idx]
#     tDA = tDA_list[idx]
#     tHB = tHB_list[idx]
#     tDB = tDB_list[idx]
#     paged_kvcache_ops.normal_h2d(tDA.data_ptr(), tHA.data_ptr(), tHA.nbytes)
#     paged_kvcache_ops.customized_h2d(tDB.data_ptr(), tHB.data_ptr(), tHB.nbytes)
#     print()

# for idx in range(0, 20):
#     tDA = tDA_list[idx]
#     tDB = tDB_list[idx]
#     paged_kvcache_ops.normal_d2d(tDA.data_ptr(), tDB.data_ptr(), tDA.nbytes)
#     print()
# torch.cuda.profiler.stop()

# for ii in range(20):
#     assert torch.allclose(tDB[ii], tHB[ii].cuda())
#     assert torch.allclose(tDB[ii], tDA[ii])


decomp_chunk_numel = 1024 * 2 * 4 * 128
decomp_chunk_bytes = decomp_chunk_numel * 2

max_out_bytes = paged_kvcache_ops.get_comp_max_chunksize(decomp_chunk_bytes)
max_out_numel = max_out_bytes // 2


f = open("/home/junyiq/newscratch/kv_dump_bf16_to_bytes_kuairank_1k/kv_seq0_layer0.log", 'r')
x = np.fromfile(f, dtype=np.int16)
# DA = torch.from_numpy(x, dtype=bfloat16).cuda().view(-1, 2 * 4 * 128)


num_comp_pages = 2

# DA = torch.rand((num_comp_pages, decomp_chunk_numel), dtype=torch.bfloat16, device=torch.cuda.current_device())
DA = torch.from_file("/home/junyiq/newscratch/kv_dump_bf16_to_bytes_kuairank_1k/kv_seq0_layer0.log", shared=False, size=8086 * 2 * 4 * 128, dtype=torch.bfloat16) 
DA = DA.clone().cuda().view(-1, )[:decomp_chunk_numel*num_comp_pages].view(-1, decomp_chunk_numel)

DB = torch.zeros((num_comp_pages, max_out_numel), dtype=torch.bfloat16, device=torch.cuda.current_device())

PA = torch.zeros((num_comp_pages,), dtype=torch.long, device=torch.cuda.current_device())
SA = torch.zeros((num_comp_pages,), dtype=torch.long, device=torch.cuda.current_device())
PB = torch.zeros((num_comp_pages,), dtype=torch.long, device=torch.cuda.current_device())
SB = torch.zeros((num_comp_pages,), dtype=torch.long, device=torch.cuda.current_device())

comp_tmp_bytes = paged_kvcache_ops.get_comp_temp_size(num_comp_pages, decomp_chunk_bytes)
comp_tmp_numel = comp_tmp_bytes // 2
comp_tmp_buffer = torch.empty((comp_tmp_numel,), dtype=torch.bfloat16, device=torch.cuda.current_device())

paged_kvcache_ops.compress(DA, PA, SA, DB, PB, SB, comp_tmp_buffer)

print(SB / SA)

DC = torch.zeros((num_comp_pages, decomp_chunk_numel), dtype=torch.bfloat16, device=torch.cuda.current_device())
PC = torch.zeros((num_comp_pages,), dtype=torch.long, device=torch.cuda.current_device())
SCin = torch.ones((num_comp_pages,), dtype=torch.long, device=torch.cuda.current_device()) # * decomp_chunk_bytes
SCout = torch.zeros((num_comp_pages,), dtype=torch.long, device=torch.cuda.current_device())
Status = torch.zeros((num_comp_pages,), dtype=torch.int, device=torch.cuda.current_device())

decomp_tmp_bytes = paged_kvcache_ops.get_decomp_temp_size(num_comp_pages, decomp_chunk_bytes)
decomp_tmp_numel = decomp_tmp_bytes // 2
decomp_tmp_buffer = torch.empty((decomp_tmp_numel,), dtype=torch.bfloat16, device=torch.cuda.current_device())

paged_kvcache_ops.decompress(DB, PB, SB, DC, PC, SCin, SCout, decomp_tmp_buffer, Status)

print(torch.max(torch.abs((DC-DA)/DA)))
print(torch.allclose(DC, DA))