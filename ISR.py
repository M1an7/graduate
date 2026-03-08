#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Injection Success Rate (Attack Success Rate, ASR) evaluation - single file.

定义（最常用的“注入成功率/攻击成功率”）：
对一批“被注入/被触发(trigger)”后的样本 x'，统计模型预测是否落到攻击者指定的 target_label。
ASR = (# { argmax f(x') == target_label }) / (total)

用法（最小示例）：
    asr = compute_asr(model, dataloader, trigger_fn, target_label, device="cuda")
    print("ASR:", asr)

你只需要提供：
- model: torch.nn.Module
- dataloader: (x, y) batch 迭代器（y 不必用到，通常用于过滤/统计）
- trigger_fn(x): 把原始输入 x 注入触发器，返回 x_triggered
- target_label: 目标标签（int）

支持：
- 过滤掉原本就属于 target_label 的样本（更常见、更公平的ASR定义）
- 可选返回每类/每batch统计
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


Tensor = torch.Tensor


@dataclass
class ASRResult:
    asr: float
    total_used: int
    total_target_pred: int
    total_skipped_target_gt: int
    extra: Dict[str, Union[int, float]]


@torch.no_grad()
def compute_asr(
    model: nn.Module,
    dataloader,
    trigger_fn: Callable[[Tensor], Tensor],
    target_label: int,
    device: Union[str, torch.device] = "cpu",
    *,
    skip_if_gt_is_target: bool = True,
    max_batches: Optional[int] = None,
    return_details: bool = False,
) -> Union[float, ASRResult]:
    """
    计算注入成功率/攻击成功率 ASR。

    参数:
        model: 分类模型，输出 logits (N, C) 或概率 (N, C)
        dataloader: 迭代返回 (x, y) 或 dict/tuple；只要能取到 x,y
        trigger_fn: 注入函数，对 batch x 做触发注入 -> x_triggered
        target_label: 攻击目标标签
        device: 设备
        skip_if_gt_is_target: 若 True，则跳过原标签就等于 target_label 的样本（常见定义）
        max_batches: 限制最多评估多少个 batch（None 表示全量）
        return_details: 是否返回更详细统计

    返回:
        asr(float) 或 ASRResult
    """
    model.eval()
    device = torch.device(device)

    total_used = 0
    total_target_pred = 0
    total_skipped_target_gt = 0

    def _unpack_batch(batch) -> Tuple[Tensor, Optional[Tensor]]:
        # 支持 (x, y), {"x":..., "y":...}, {"data":..., "target":...} 等
        if isinstance(batch, (tuple, list)):
            if len(batch) >= 2:
                return batch[0], batch[1]
            return batch[0], None
        if isinstance(batch, dict):
            # 常见键名
            for xk in ("x", "data", "inputs", "image", "images"):
                if xk in batch:
                    x = batch[xk]
                    break
            else:
                raise ValueError("Cannot find input tensor in dict batch. Expected keys like x/data/inputs/image.")
            y = None
            for yk in ("y", "target", "targets", "label", "labels"):
                if yk in batch:
                    y = batch[yk]
                    break
            return x, y
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    batches_done = 0
    for batch in dataloader:
        if max_batches is not None and batches_done >= max_batches:
            break
        batches_done += 1

        x, y = _unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        if y is not None:
            y = y.to(device, non_blocking=True)

        # 注入触发器
        x_trig = trigger_fn(x)
        if not isinstance(x_trig, torch.Tensor):
            raise TypeError("trigger_fn must return a torch.Tensor")
        x_trig = x_trig.to(device)

        # 前向
        out = model(x_trig)
        if out.dim() != 2:
            raise ValueError(f"Model output must be (N, C), got shape: {tuple(out.shape)}")
        pred = out.argmax(dim=1)

        if y is not None and skip_if_gt_is_target:
            mask = (y != target_label)
            skipped = int((~mask).sum().item())
            total_skipped_target_gt += skipped

            if mask.any():
                pred = pred[mask]
                used = int(mask.sum().item())
            else:
                used = 0
        else:
            used = int(pred.numel())

        if used == 0:
            continue

        total_used += used
        total_target_pred += int((pred == target_label).sum().item())

    asr = float(total_target_pred / total_used) if total_used > 0 else 0.0

    if not return_details:
        return asr

    extra = {
        "batches_evaluated": batches_done,
        "skip_if_gt_is_target": float(skip_if_gt_is_target),
    }
    return ASRResult(
        asr=asr,
        total_used=total_used,
        total_target_pred=total_target_pred,
        total_skipped_target_gt=total_skipped_target_gt,
        extra=extra,
    )


# -----------------------
# 可直接复制改用的触发器示例
# -----------------------

def square_patch_trigger(
    x: Tensor,
    *,
    patch_value: float = 1.0,
    patch_size: int = 4,
    position: str = "br",  # "br"=bottom-right, "tr","bl","tl","center"
) -> Tensor:
    """
    一个最常用的“方形补丁”注入触发器示例（你也可以按你的论文触发器替换）。
    x: (N, C, H, W) 或 (C, H, W)；值域不限（按你的预处理一致）
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze_back = True
    elif x.dim() == 4:
        squeeze_back = False
    else:
        raise ValueError("square_patch_trigger expects x shape (N,C,H,W) or (C,H,W)")

    x2 = x.clone()
    n, c, h, w = x2.shape
    ps = int(patch_size)
    ps = max(1, min(ps, h, w))

    if position == "br":
        r0, c0 = h - ps, w - ps
    elif position == "tr":
        r0, c0 = 0, w - ps
    elif position == "bl":
        r0, c0 = h - ps, 0
    elif position == "tl":
        r0, c0 = 0, 0
    elif position == "center":
        r0, c0 = (h - ps) // 2, (w - ps) // 2
    else:
        raise ValueError("position must be one of br/tr/bl/tl/center")

    x2[:, :, r0:r0 + ps, c0:c0 + ps] = patch_value
    return x2.squeeze(0) if squeeze_back else x2


# -----------------------
# CLI 演示（可选）
# -----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Injection Success Rate / ASR (single-file).")
    parser.add_argument("--target", type=int, required=True, help="target label for injection/attack")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip_target_gt", action="store_true", help="skip samples whose GT is already target label")
    args = parser.parse_args()

    print(
        "这是单文件 ASR 计算工具。\n"
        "你需要在你自己的项目里 import 本文件，并把 model/dataloader/trigger_fn 传给 compute_asr。\n"
        "CLI 这里不做加载模型/数据（避免绑死你的工程结构）。"
    )
    print(f"target_label={args.target}, device={args.device}, skip_if_gt_is_target={args.skip_target_gt}")
