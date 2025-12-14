# test.py
import os
import torch
import argparse
import torch.distributed as dist
from torch.distributed._tensor import DTensor, DeviceMesh
from torch.distributed._tensor.placement_types import Replicate

# torchrun --nproc_per_node=2 dtensor_to_tensor.py

def state_dict_to_cpu(sd, mesh):
    new_sd = {}
    for k, v in sd.items():
        if isinstance(v, DTensor):
            # all‑gather the full tensor onto each GPU
            v = v.redistribute(device_mesh=mesh, placements=[Replicate()])
            new_sd[k] = v.to_local().detach().cpu()
        elif isinstance(v, torch.Tensor):
            new_sd[k] = v.detach().cpu()
        else:
            new_sd[k] = v
    return new_sd

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DTensor conversion")

    parser = argparse.ArgumentParser(description="Convert DTensor state dict to CPU tensors")
    parser.add_argument("--dtensor_dir", type=str, required=True, help="Path to the Hugging Face directory containing the model files")
    args = parser.parse_args()
    # 1) figure out which GPU this process should use
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # 2) initialize the NCCL process group
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()

    # 3) rebuild your mesh
    mesh = DeviceMesh("cuda", list(range(world_size)))

    # 4) load the checkpoint so that each local shard lives on its GPU
    dtensor_dir = args.dtensor_dir
    if not os.path.isdir(dtensor_dir):
        raise FileNotFoundError(f"Hugging Face directory not found: {dtensor_dir}")
    if not os.path.exists(os.path.join(dtensor_dir, "evidential_head_dtensor.pth")) or not os.path.exists(os.path.join(dtensor_dir, "cls_head_dtensor.pth")):
        raise FileNotFoundError("Required head module files not found in the specified directory")
    # Load the state dicts from the specified files
    map_loc = f"cuda:{local_rank}"
    evi_sd = torch.load(f"{dtensor_dir}/evidential_head_dtensor.pth", map_location=map_loc)
    cls_sd = torch.load(f"{dtensor_dir}/cls_head_dtensor.pth", map_location=map_loc)

    # 5) redistribute→gather→to_cpu
    evi_sd_head = state_dict_to_cpu(evi_sd, mesh)
    cls_sd_head = state_dict_to_cpu(cls_sd, mesh)

    if dist.get_rank() == 0:
        print(evi_sd_head)
        print("weight shape:", evi_sd_head["weight"].shape)
        print(cls_sd_head)
        print("weight shape:", cls_sd_head["weight"].shape)

    out_path = os.path.join(dtensor_dir, "evidential_head_tensor.pth")
    torch.save(evi_sd_head, out_path)

    out_path = os.path.join(dtensor_dir, "cls_head_tensor.pth")
    torch.save(cls_sd_head, out_path)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

