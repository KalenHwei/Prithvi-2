# MoE.py
import math
from typing import List, Tuple, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FFN",
    "MoEMLP",
    "inject_moe",
    "collect_moe_aux_loss",
    "reset_moe_aux_loss",
    "patch_training_step_with_moe",
]

# FFN For Expert use
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N, D]
        return self.fc2(self.dropout(self.act(self.fc1(x))))


# MoEMLP: MoE-MLP 核心
class MoEMLP(nn.Module):
    """
    稀疏激活 MoE 前馈层（替换 Transformer/ViT 的 MLP/FFN）
    参数:
      d_model: 隐层维度
      d_ff: FFN内部扩展维度
      n_experts: 专家数
      top_k: 1(快) 或 2(更稳)
      capacity_factor: 每个专家容量放大系数，1.0~1.25 常用
      dropout: Expert FFN's dropout
      normalize_router_logits: 对 router logits 做归一化（可改善稳定性）
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 4,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        dropout: float = 0.0,
        normalize_router_logits: bool = False,
    ):
        super().__init__()
        assert top_k in (1, 2) # 限制死1和2个专家
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.normalize_router_logits = normalize_router_logits

        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList(
            [FFN(d_model, d_ff, dropout=dropout) for _ in range(n_experts)]
        )

        # 运行期记录 aux loss（便于 Lightning 汇总）
        self._last_aux_loss: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _compute_capacity(self, num_tokens: int) -> int:
        avg = num_tokens / self.n_experts
        cap = int(self.capacity_factor * avg)
        return max(1, cap)

    def _load_balancing_loss(
        self,
        gates_softmax: torch.Tensor,   # [N, E]
        hard_assign_mask: torch.Tensor # [N, E] (one-hot 或 two-hot)
    ) -> torch.Tensor:
        N, E = gates_softmax.shape
        importance = gates_softmax.mean(dim=0)       # soft 概率均值
        load = hard_assign_mask.float().mean(dim=0)  # 硬分配均值
        return E * (importance * load).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        返回: y: [B, T, D]
        说明: aux loss 存到 self._last_aux_loss，外部可读取加到总损失
        """
        B, T, D = x.shape
        N = B * T
        dtype = x.dtype
        device = x.device

        # Router（建议 fp32 以稳）
        logits = self.router(x.reshape(N, D).float())  # [N, E]
        if self.normalize_router_logits:
            logits = logits / (logits.std(dim=-1, keepdim=True) + 1e-6)
        gates = F.softmax(logits, dim=-1)              # [N, E]
        topk_vals, topk_idx = torch.topk(gates, k=self.top_k, dim=-1)  # [N, k], [N, k]

        # 硬分配 mask（用于 aux loss）
        hard_mask = torch.zeros_like(gates, dtype=torch.bool)
        hard_mask.scatter_(dim=-1, index=topk_idx, src=torch.ones_like(topk_idx, dtype=torch.bool))

        # 容量限制
        capacity = self._compute_capacity(N)

        # 将 token 分桶到各 expert（简洁实现，便于理解；性能可再向量化/EP优化）
        x_flat = x.reshape(N, D)
        expert_token_ids: List[List[int]] = [[] for _ in range(self.n_experts)]
        expert_gate_vals: List[List[float]] = [[] for _ in range(self.n_experts)]
        for i in range(N):
            for j in range(self.top_k):
                e = int(topk_idx[i, j])
                expert_token_ids[e].append(i)
                expert_gate_vals[e].append(float(topk_vals[i, j]))

        for e in range(self.n_experts):
            if len(expert_token_ids[e]) > capacity:
                expert_token_ids[e] = expert_token_ids[e][:capacity]
                expert_gate_vals[e] = expert_gate_vals[e][:capacity]

        # Experts 前向
        expert_outputs: List[torch.Tensor] = []
        metas: List[Tuple[int, int, float]] = []  # (expert_id, token_id, gate)
        for e in range(self.n_experts):
            ids = expert_token_ids[e]
            if len(ids) == 0:
                expert_outputs.append(x_flat.new_zeros(0, D))
            else:
                inp = x_flat[torch.tensor(ids, device=device, dtype=torch.long)]
                out = self.experts[e](inp.to(dtype))            # 让专家在主 dtype 上算
                expert_outputs.append(out)
                metas.extend((e, tid, g) for tid, g in zip(ids, expert_gate_vals[e]))

        # 合并回原顺序（Top-2 可能累加）
        y_flat = x_flat.new_zeros(N, D)
        wsum = x_flat.new_zeros(N, 1)
        offset = 0
        for e in range(self.n_experts):
            n_e = expert_outputs[e].shape[0]
            if n_e == 0:
                continue
            slice_meta = metas[offset: offset + n_e]
            out = expert_outputs[e]
            for row, (_, tok, g) in enumerate(slice_meta):
                y_flat[tok] += g * out[row]
                wsum[tok, 0] += g
            offset += n_e

        y_flat = torch.where(wsum > 0, y_flat / wsum.clamp(min=1e-6), y_flat)
        y = y_flat.reshape(B, T, D).to(dtype)

        # 记录 aux loss（外部自行乘以系数）
        self._last_aux_loss = self._load_balancing_loss(gates, hard_mask).to(dtype).to(device)
        return y


# --------------------------
# 工具：自动替换 MLP->MoE
# --------------------------
def _looks_like_mlp(module: nn.Module) -> bool:
    """
    经验性判断：是否像 ViT 的 MLP/FFN 模块
    - 含 fc1/fc2（Linear）
    - 或 子模块名包含 'mlp'/'ffn'
    """
    # 直接命名命中
    name = module.__class__.__name__.lower()
    if "mlp" in name or "ffn" in name:
        return True

    # 结构命中：有两个线性层 fc1/fc2
    if hasattr(module, "fc1") and hasattr(module, "fc2"):
        if isinstance(module.fc1, nn.Linear) and isinstance(module.fc2, nn.Linear):
            return True
    return False


def _make_moe_from_mlp(mlp_module: nn.Module, n_experts: int, top_k: int,
                       capacity_factor: float, dropout: float) -> MoEMLP:
    # 推断维度
    if hasattr(mlp_module, "fc1") and hasattr(mlp_module, "fc2"):
        d_model = mlp_module.fc1.in_features
        d_ff = mlp_module.fc1.out_features
    else:
        # 回退：尝试从子模块里找两个 Linear
        linears = [m for m in mlp_module.modules() if isinstance(m, nn.Linear)]
        assert len(linears) >= 2, "Cannot infer dimensions for MoE; please pass custom builder."
        d_model = linears[0].in_features
        d_ff = linears[0].out_features

    moe = MoEMLP(
        d_model=d_model,
        d_ff=d_ff,
        n_experts=n_experts,
        top_k=top_k,
        capacity_factor=capacity_factor,
        dropout=dropout,
    )

    # 将原 mlp 的权重迁移到第 0 号 expert（可选，便于 warm-start）
    if hasattr(mlp_module, "fc1") and hasattr(mlp_module, "fc2"):
        try:
            moe.experts[0].fc1.weight.data.copy_(mlp_module.fc1.weight.data)
            moe.experts[0].fc1.bias.data.copy_(mlp_module.fc1.bias.data)
            moe.experts[0].fc2.weight.data.copy_(mlp_module.fc2.weight.data)
            moe.experts[0].fc2.bias.data.copy_(mlp_module.fc2.bias.data)
        except Exception:
            pass
    return moe


def inject_moe(
    root: nn.Module,
    n_experts: int = 4,
    top_k: int = 1,
    capacity_factor: float = 1.25,
    dropout: float = 0.0,
    select: str = "last_k",   # "all" | "last_k"
    k: int = 4,
    name_keywords: Iterable[str] = ("mlp", "ffn"),
) -> List[str]:
    """
    递归替换模型中的 MLP/FFN 子层为 MoEMLP。

    返回：被替换层的完全限定名列表。
    """
    replaced: List[str] = []
    candidates: List[str] = []

    # 收集候选
    for full_name, module in root.named_modules():
        low = full_name.lower()
        if any(kw in low for kw in name_keywords) or _looks_like_mlp(module):
            # 排除 MoE 自己
            if isinstance(module, MoEMLP):
                continue
            # 过滤掉 FFN 的内部 Linear 本体，只针对上层模块
            candidates.append(full_name)

    if not candidates:
        print("[MoE] No MLP/FFN candidates found. "
              "Consider adjusting name_keywords or implementing a custom selector.")
        return replaced

    # 选择策略
    if select == "all":
        targets = candidates
    elif select == "last_k":
        targets = candidates[-k:] if k > 0 else []
    else:
        raise ValueError(f"Unknown select strategy: {select}")

    # 执行替换
    for full_name in targets:
        parent, attr = _resolve_parent_and_attr(root, full_name)
        old = getattr(parent, attr)
        moe = _make_moe_from_mlp(old, n_experts, top_k, capacity_factor, dropout)
        setattr(parent, attr, moe)
        replaced.append(full_name)

    print(f"[MoE] Replaced {len(replaced)} layer(s):")
    for n in replaced:
        print("   -", n)
    return replaced


def _resolve_parent_and_attr(root: nn.Module, full_name: str):
    """
    根据完全限定名找到父模块与属性名，便于 setattr 替换。
    """
    parts = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def collect_moe_aux_loss(root: nn.Module) -> torch.Tensor:
    aux = None
    for m in root.modules():
        if isinstance(m, MoEMLP) and (m._last_aux_loss is not None):
            aux = m._last_aux_loss if aux is None else aux + m._last_aux_loss
    if aux is None:
        # 没有 MoE 或尚未前向
        return torch.tensor(0.0, device=next(root.parameters()).device)
    return aux

def reset_moe_aux_loss(root: nn.Module):
    for m in root.modules():
        if isinstance(m, MoEMLP):
            m._last_aux_loss = None


def patch_training_step_with_moe(lightning_module: "nn.Module", aux_coef: float = 1e-2):
    """
    包装原 training_step，在返回 loss 之前把 sum(moe_aux_loss)*aux_coef 加上。
    兼容 training_step 返回 Tensor 或 dict 的常见写法。
    """
    if not hasattr(lightning_module, "training_step"):
        raise AttributeError("The given module has no training_step to patch.")

    orig_training_step = lightning_module.training_step

    def wrapped_training_step(*args, **kwargs):
        result = orig_training_step(*args, **kwargs)

        # 收集 MoE aux loss
        aux = collect_moe_aux_loss(lightning_module) * aux_coef
        reset_moe_aux_loss(lightning_module)

        # 两种常见返回：Tensor 或 dict
        if torch.is_tensor(result):
            return result + aux
        if isinstance(result, dict):
            if "loss" in result and torch.is_tensor(result["loss"]):
                result["loss"] = result["loss"] + aux
                # 记录日志键
                if "moe_aux" not in result:
                    result["moe_aux"] = aux.detach()
                return result
        # 其他自定义情况：直接返回原值（你也可以在此处 raise 提醒）
        return result

    lightning_module.training_step = wrapped_training_step
    print(f"[MoE] Patched training_step with aux_coef={aux_coef}.")
