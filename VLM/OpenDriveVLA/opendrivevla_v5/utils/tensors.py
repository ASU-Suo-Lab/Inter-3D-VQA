from __future__ import annotations

import torch
from mmcv.parallel import DataContainer


def remove_datacontainer(data):
    if isinstance(data, DataContainer):
        return remove_datacontainer(data.data)
    if isinstance(data, dict):
        return {key: remove_datacontainer(value) for key, value in data.items()}
    if isinstance(data, list):
        return [remove_datacontainer(item) for item in data]
    if isinstance(data, tuple):
        return tuple(remove_datacontainer(item) for item in data)
    return data


def move_data_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {key: move_data_to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [move_data_to_device(item, device) for item in data]
    if isinstance(data, tuple):
        return tuple(move_data_to_device(item, device) for item in data)
    return data


def change_tensor_to_float16(data):
    if isinstance(data, torch.Tensor):
        if data.dtype in [torch.float32, torch.float64]:
            return data.to(dtype=torch.float16)
        return data
    if isinstance(data, dict):
        return {key: change_tensor_to_float16(value) for key, value in data.items()}
    if isinstance(data, list):
        return [change_tensor_to_float16(item) for item in data]
    if isinstance(data, tuple):
        return tuple(change_tensor_to_float16(item) for item in data)
    return data


def change_tensor_to_bfloat16(data):
    if isinstance(data, torch.Tensor):
        if data.dtype in [torch.float16, torch.float32, torch.float64]:
            return data.to(dtype=torch.bfloat16)
        return data
    if isinstance(data, dict):
        return {key: change_tensor_to_bfloat16(value) for key, value in data.items()}
    if isinstance(data, list):
        return [change_tensor_to_bfloat16(item) for item in data]
    if isinstance(data, tuple):
        return tuple(change_tensor_to_bfloat16(item) for item in data)
    return data


def change_tensor_to_float32(data):
    if isinstance(data, torch.Tensor):
        if data.dtype in [torch.float16, torch.bfloat16]:
            return data.to(dtype=torch.float32)
        return data
    if isinstance(data, dict):
        return {key: change_tensor_to_float32(value) for key, value in data.items()}
    if isinstance(data, list):
        return [change_tensor_to_float32(item) for item in data]
    if isinstance(data, tuple):
        return tuple(change_tensor_to_float32(item) for item in data)
    return data
