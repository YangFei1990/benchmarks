import torch
import torch.distributed as dist
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--arg1", type=int, default=0, help="num_microbatches")
parser.add_argument("--arg2", type=int, default=1, help="tensor_parallel_size")
args, _ = parser.parse_known_args()
print(f"arg1 {args.arg1} arg2 {args.arg2}")
print("init process group")
dist.init_process_group("nccl")
print("rank:", dist.get_rank())
torch.cuda.set_device(dist.get_rank() % 8)
tensor = torch.randn(4,4, device="cuda")
print(f"[{dist.get_rank()}] tensor {tensor}")
dist.all_reduce(tensor)
print(f"[{dist.get_rank()}] tensor {tensor} after reduce")
