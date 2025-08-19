# MoE.py
import math
from typing import List, Tuple, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
# 假设在 Pytorch Lightning 环境下
try:
    import lightning.pytorch as pl
except ImportError:
    pl = None


__all__ = [
    "DeepSeekMoELayer",
    "inject_moe",
    "collect_moe_aux_loss",
    "reset_moe_aux_loss",
    "patch_training_step_with_moe",
]

# --- 核心模块 (类 DeepSeek-MoE 架构) ---

class DeepSeekMoELayer(nn.Module):
    """
    一个高性能、向量化的 MoE 层，其架构类似于 DeepSeek-MoE。
    特点是共享的上半部分 FFN (up-projection) 和专家化的下半部分 FFN (down-projection)。

    参数:
      d_model (int): 输入和输出的维度。
      d_hidden (int): FFN 内部的隐藏层维度。
      n_experts (int): 专家网络的数量。
      top_k (int): 每个 token 选择得分最高的 k 个专家。通常为 1 或 2。
      capacity_factor (float): 容量因子，用于计算每个专家的缓冲区大小。
      dropout (float): FFN 内部的 dropout 比率。
      normalize_router_logits (bool): 是否对路由器的 logits 进行标准化。
      router_z_loss_coef (float): 路由器 Z 损失的系数。
    """
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        n_experts: int = 4,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        dropout: float = 0.0,
        normalize_router_logits: bool = False,
        router_z_loss_coef: float = 0.001,
    ):
        super().__init__()
        assert top_k in (1, 2), "目前仅支持 top_k=1 或 2"
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.normalize_router_logits = normalize_router_logits
        self.router_z_loss_coef = router_z_loss_coef

        # 路由器
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # 共享的 FFN 上半部分 (Up-Projection)
        self.gate_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.act = nn.SiLU()  # SwiGLU/SiLU 在现代模型中更常见
        self.dropout = nn.Dropout(dropout)

        # 专家化的 FFN 下半部分 (Down-Projection)
        self.down_projs = nn.ModuleList(
            [nn.Linear(d_hidden, d_model, bias=False) for _ in range(n_experts)]
        )

        # 运行时记录辅助损失
        self._last_aux_loss: Optional[torch.Tensor] = None

    def _compute_capacity(self, num_tokens: int) -> int:
        """计算每个专家的容量。"""
        return math.ceil((num_tokens / self.n_experts) * self.capacity_factor)

    def _load_balancing_loss(self, gates_softmax: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        """计算负载均衡损失。"""
        N, E = gates_softmax.shape
        load = expert_mask.float().sum(dim=0) / N
        importance = gates_softmax.sum(dim=0) / N
        return self.n_experts * torch.sum(load * importance)

    def _router_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """计算 Router Z-Loss，鼓励 logits 的量级。"""
        if self.router_z_loss_coef > 0:
            log_z = torch.logsumexp(router_logits, dim=-1)
            return torch.mean(log_z**2) * self.router_z_loss_coef
        return torch.tensor(0.0, device=router_logits.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        高性能的向量化前向传播。
        输入 x: [B, T, D]
        返回 y: [B, T, D]
        """
        original_shape = x.shape
        B, T, D = original_shape
        N = B * T

        x_flat = x.reshape(N, D)

        # 1. 路由器计算 & Top-k 门控
        logits = self.router(x_flat.float())
        if self.normalize_router_logits:
            logits = logits / (logits.std(dim=-1, keepdim=True) + 1e-6)

        gates = F.softmax(logits, dim=-1)
        topk_gates, topk_indices = torch.topk(gates, k=self.top_k, dim=-1)

        # 2. 计算辅助损失
        expert_mask = F.one_hot(topk_indices, self.n_experts).sum(dim=1)
        load_balancing_loss = self._load_balancing_loss(gates, expert_mask)
        router_z_loss = self._router_z_loss(logits)
        self._last_aux_loss = load_balancing_loss + router_z_loss

        # 3. 容量限制 & Token 调度
        capacity = self._compute_capacity(N)
        position_in_expert = torch.cumsum(expert_mask, dim=0) - 1
        expert_mask = expert_mask * (position_in_expert < capacity)
        # 将被丢弃的 token 的门控值清零
        topk_gates = topk_gates * (expert_mask.gather(dim=1, index=topk_indices) > 0)
        # 重新归一化门控值 (可选但推荐)
        topk_gates = topk_gates / (topk_gates.sum(dim=-1, keepdim=True) + 1e-6)

        # 4. 共享 FFN 上半部分计算
        hidden_states = self.act(self.gate_proj(x_flat))
        hidden_states = self.dropout(hidden_states)

        # 5. 向量化专家计算
        y_flat = torch.zeros_like(x_flat)
        # 将 top-k 个门控值和索引展平，以便进行高效的 index_add_ 操作
        flat_topk_indices = topk_indices.flatten()
        flat_topk_gates = topk_gates.flatten()
        # 创建一个 token 索引，用于将 hidden_states 与展平的门控值对齐
        token_source_indices = torch.arange(N, device=x.device).repeat_interleave(self.top_k)

        # 遍历每个专家
        for i in range(self.n_experts):
            # 找到分配给当前专家的 token
            expert_mask_i = (flat_topk_indices == i)
            if expert_mask_i.any():
                # 获取这些 token 的索引和门控值
                token_indices_i = token_source_indices[expert_mask_i]
                gates_i = flat_topk_gates[expert_mask_i].unsqueeze(1)
                
                # 从 hidden_states 中收集输入
                expert_inputs = hidden_states[token_indices_i]
                
                # 运行专家网络 (只有 down_proj)
                expert_outputs = self.down_projs[i](expert_inputs)
                
                # 将加权后的输出添加回最终结果
                y_flat.index_add_(0, token_indices_i, expert_outputs * gates_i)

        return y_flat.reshape(original_shape)

# --- 自动替换工具 (已更新以支持新架构) ---

def _looks_like_mlp(module: nn.Module) -> bool:
    """启发式地判断一个模块是否是 MLP/FFN。"""
    name = module.__class__.__name__.lower()
    if "mlp" in name or "ffn" in name or "feedforward" in name:
        return True
    # 适用于 huggingface transformers 的 ViT/BERT 等模型
    if hasattr(module, "dense") and hasattr(module, "intermediate"):
        return True
    # 适用于您之前的 FFN 定义
    if hasattr(module, "fc1") and hasattr(module, "fc2"):
        return isinstance(module.fc1, nn.Linear) and isinstance(module.fc2, nn.Linear)
    return False

def _make_moe_from_mlp(
    mlp_module: nn.Module, n_experts: int, top_k: int,
    capacity_factor: float, dropout: float
) -> DeepSeekMoELayer:
    """根据一个已有的 MLP 模块创建并初始化一个 DeepSeekMoELayer。"""
    d_model, d_hidden = -1, -1

    # 尝试多种方式推断维度
    if hasattr(mlp_module, "fc1"): # from your FFN
        up_proj, down_proj = mlp_module.fc1, mlp_module.fc2
    elif hasattr(mlp_module, "intermediate") and hasattr(mlp_module, "output"): # from BERT
        up_proj, down_proj = mlp_module.intermediate.dense, mlp_module.output.dense
    elif hasattr(mlp_module, "gate_proj") and hasattr(mlp_module, "down_proj"): # from Llama
        up_proj, down_proj = mlp_module.gate_proj, mlp_module.down_proj
    else:
        linears = [m for m in mlp_module.modules() if isinstance(m, nn.Linear)]
        if len(linears) >= 2:
            up_proj, down_proj = linears[0], linears[1]
        else:
            raise AssertionError("无法自动推断MoE维度；请传递自定义的构建器。")

    d_model = down_proj.out_features
    d_hidden = down_proj.in_features
    
    moe = DeepSeekMoELayer(
        d_model=d_model, d_hidden=d_hidden, n_experts=n_experts,
        top_k=top_k, capacity_factor=capacity_factor, dropout=dropout,
    )

    # 将原 MLP 的权重迁移到共享层和第 0 号专家，便于热启动
    try:
        moe.gate_proj.load_state_dict(up_proj.state_dict())
        moe.down_projs[0].load_state_dict(down_proj.state_dict())
        print(f"[MoE] Successfully copied weights from original MLP to MoE shared gate and expert 0.")
    except Exception as e:
        print(f"[MoE] Warning: Failed to copy weights. Reason: {e}")

    return moe

def _resolve_parent_and_attr(root: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    """根据模块的完整名称找到其父模块和它在父模块中的属性名。"""
    parts = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def inject_moe(
    root: nn.Module,
    n_experts: int = 4,
    top_k: int = 1,
    capacity_factor: float = 1.25,
    dropout: float = 0.0,
    select: str = "last_k",
    k: int = 4,
    name_keywords: Iterable[str] = ("mlp", "ffn", "feedforward"),
) -> int:
    """
    递归地将模型中的 MLP/FFN 子模块替换为 DeepSeekMoELayer。
    返回: (int) 被成功替换的层的数量。
    """
    candidates = set()
    for full_name, module in root.named_modules():
        if not any(isinstance(c, DeepSeekMoELayer) for c in module.children()):
             if _looks_like_mlp(module) or any(kw in full_name.lower() for kw in name_keywords):
                is_submodule = any(full_name.startswith(p + ".") for p in candidates)
                if not is_submodule:
                    candidates.add(full_name)

    if not candidates:
        print("[MoE] No MLP/FFN candidates found to replace.")
        return 0
    
    sorted_candidates = sorted(list(candidates))

    if select == "all":
        targets = sorted_candidates
    elif select == "last_k":
        targets = sorted_candidates[-k:] if k > 0 else []
    else:
        raise ValueError(f"Unknown select strategy: {select}")

    replaced_count = 0
    print(f"[MoE] Found {len(targets)} layer(s) to replace:")
    for full_name in targets:
        try:
            parent, attr = _resolve_parent_and_attr(root, full_name)
            old_module = getattr(parent, attr)
            moe_module = _make_moe_from_mlp(old_module, n_experts, top_k, capacity_factor, dropout)
            setattr(parent, attr, moe_module)
            print(f"   - Replaced '{full_name}' with DeepSeekMoELayer.")
            replaced_count += 1
        except Exception as e:
            print(f"   - Failed to replace '{full_name}'. Reason: {e}")

    return replaced_count


# --- PyTorch Lightning 集成工具 (无需更改) ---

def collect_moe_aux_loss(root: nn.Module) -> torch.Tensor:
    """从模型的所有 MoE 子模块中收集辅助损失并求和。"""
    aux_loss_sum = torch.tensor(0.0, device=next(root.parameters()).device)
    for m in root.modules():
        if isinstance(m, DeepSeekMoELayer) and m._last_aux_loss is not None:
            aux_loss_sum += m._last_aux_loss
    return aux_loss_sum

def reset_moe_aux_loss(root: nn.Module):
    """重置所有 MoE 子模块中缓存的辅助损失。"""
    for m in root.modules():
        if isinstance(m, DeepSeekMoELayer):
            m._last_aux_loss = None

def patch_training_step_with_moe(lightning_module: "pl.LightningModule", aux_coef: float = 1e-2):
    """通过 Monkey Patching 自动将 MoE 辅助损失添加到总损失中。"""
    if pl is None:
        raise ImportError("`lightning.pytorch` is required for this function.")
    if not hasattr(lightning_module, "training_step"):
        raise AttributeError("给定的模块没有 training_step 方法可以修补。")

    original_step = lightning_module.training_step

    def patched_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        aux_loss = collect_moe_aux_loss(lightning_module) * aux_coef
        reset_moe_aux_loss(lightning_module)

        if isinstance(result, torch.Tensor):
            return result + aux_loss
        elif isinstance(result, dict) and "loss" in result:
            result["loss"] = result["loss"] + aux_loss
            lightning_module.log("train/moe_aux_loss", aux_loss.detach(), on_step=True, on_epoch=False, prog_bar=True)
            return result
        else:
            print("[MoE Patch] Warning: Could not add aux_loss to training_step result.")
            return result
    
    lightning_module.training_step = patched_step
    print(f"[MoE] Patched `training_step` with aux_coef={aux_coef}.")