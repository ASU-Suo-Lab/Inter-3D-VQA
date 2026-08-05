import os
import numpy as np

import torch.distributed as dist

try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:
    VideoReader = None
    cpu = None


def process_video_with_decord(video_file, data_args):
    if VideoReader is None or cpu is None:
        raise ImportError("decord is required for video processing but is not installed.")
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    # avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
    
    if total_frame_num <= data_args.frames_upbound or data_args.frames_upbound <= 0:
        frame_idx = list(range(total_frame_num))
    else:
        frame_idx = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int).tolist()
    
    frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time_norm = [i / total_frame_num for i in frame_idx]
    
    video = vr.get_batch(frame_idx).asnumpy()
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    
    num_frames_to_sample = num_frames = len(frame_idx)
    vr.seek(0)
    
    return video, video_time, frame_time, num_frames_to_sample, frame_idx, frame_time_norm

def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
