# MoE.py
import math
from typing import List, Tuple, Iterable, Optional, Callable

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

# --- 核心模块 ---

class FFN(nn.Module):
    """一个标准的前馈网络（Feed-Forward Network），用作 MoE 中的专家模型。"""
    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。输入形状: [..., d_model]"""
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class MoEMLP(nn.Module):
    """
    一个高性能、向量化的稀疏激活混合专家（Mixture-of-Experts）层。
    它用于替换 Transformer 或 ViT 模型中的标准 MLP/FFN 层。

    参数:
      d_model (int): 输入和输出的维度。
      d_hidden (int): 专家网络内部的隐藏层维度。
      n_experts (int): 专家网络的数量。
      top_k (int): 每个 token 选择得分最高的 k 个专家。通常为 1 或 2。
      capacity_factor (float): 容量因子。用于计算每个专家的缓冲区大小，防止过载。
      dropout (float): 专家网络 FFN 中的 dropout 比率。
      normalize_router_logits (bool): 是否对路由器的 logits 进行标准化，可以提高训练稳定性。
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
    ):
        super().__init__()
        assert top_k in (1, 2), "目前仅支持 top_k=1 或 2"
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.normalize_router_logits = normalize_router_logits

        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList(
            [FFN(d_model, d_hidden, dropout=dropout) for _ in range(n_experts)]
        )
        
        # 运行时记录辅助损失，以便于 Lightning 模块在外部汇总
        self._last_aux_loss: Optional[torch.Tensor] = None

    def _compute_capacity(self, num_tokens: int) -> int:
        """计算每个专家的容量。"""
        # 每个专家的容量 = (token总数 / 专家数) * 容量因子
        capacity = math.ceil((num_tokens / self.n_experts) * self.capacity_factor)
        return max(1, capacity)

    def _load_balancing_loss(
        self,
        gates_softmax: torch.Tensor,   # [N, E]
        expert_mask: torch.Tensor      # [N, E] (one-hot 或 two-hot)
    ) -> torch.Tensor:
        """
        计算负载均衡辅助损失（load balancing auxiliary loss）。
        这个损失函数鼓励路由器将 token 均匀地分配给所有专家。
        """
        N, E = gates_softmax.shape
        # 每个专家收到的 token 比例（基于硬分配）
        load = expert_mask.float().sum(dim=0) / N
        # 路由器分配给每个专家的平均概率（基于软分配）
        importance = gates_softmax.sum(dim=0) / N
        
        # 损失 = E * dot_product(load, importance)
        return self.n_experts * torch.sum(load * importance)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        高性能的向量化前向传播。
        输入 x: [B, T, D] (批次大小, 序列长度, 模型维度)
        返回 y: [B, T, D]
        辅助损失会存储在 self._last_aux_loss 中。
        """
        original_shape = x.shape
        B, T, D = original_shape
        N = B * T  # Token 总数
        
        x_flat = x.reshape(N, D)

        # 1. 路由器计算 & Top-k 门控
        # 为了稳定性，建议在 float32 类型上计算路由器逻辑
        logits = self.router(x_flat.float())
        if self.normalize_router_logits:
            logits = logits / (logits.std(dim=-1, keepdim=True) + 1e-6)
        
        gates = F.softmax(logits, dim=-1)  # [N, E] (软门控值)
        topk_gates, topk_indices = torch.topk(gates, k=self.top_k, dim=-1)  # [N, k]
        
        # 2. 计算辅助损失
        # 创建一个硬分配的掩码，用于计算负载均衡损失
        expert_mask = F.one_hot(topk_indices, self.n_experts).sum(dim=1)  # [N, E]
        self._last_aux_loss = self._load_balancing_loss(gates, expert_mask)

        # 3. 专家容量限制 & Token 调度
        capacity = self._compute_capacity(N)
        
        # 计算每个 token 在其被分配的专家队列中的位置
        # position_in_expert = (expert_mask.cumsum(dim=0) - 1) * expert_mask
        
        # 为每个 token 计算其在专家队列中的位置
        position_in_expert = torch.cumsum(expert_mask, dim=0) - 1
        
        # 丢弃超出容量的 token
        expert_mask = expert_mask * (position_in_expert < capacity)
        topk_gates = topk_gates * (expert_mask.gather(dim=1, index=topk_indices) > 0)
        
        # 4. 向量化专家计算
        y_flat = torch.zeros_like(x_flat)
        # 遍历每个专家（循环次数少，性能影响小）
        for i in range(self.n_experts):
            # 找到分配给当前专家的 token
            expert_token_indices = torch.where(expert_mask[:, i] > 0)[0]
            if expert_token_indices.numel() == 0:
                continue

            # 从原始输入中收集这些 token
            expert_inputs = x_flat[expert_token_indices]
            
            # 运行专家网络
            expert_outputs = self.experts[i](expert_inputs)
            
            # 获取对应的门控值
            gate_values = gates[expert_token_indices, i].unsqueeze(1)
            
            # 将加权后的输出添加回最终结果
            y_flat.index_add_(0, expert_token_indices, expert_outputs * gate_values)

        # 5. 结果重塑
        return y_flat.reshape(original_shape)


# --- 自动替换工具 ---

def _looks_like_mlp(module: nn.Module) -> bool:
    """启发式地判断一个模块是否是 MLP/FFN。"""
    name = module.__class__.__name__.lower()
    if "mlp" in name or "ffn" in name:
        return True
    if hasattr(module, "fc1") and hasattr(module, "fc2"):
        if isinstance(module.fc1, nn.Linear) and isinstance(module.fc2, nn.Linear):
            return True
    return False

def _make_moe_from_mlp(
    mlp_module: nn.Module, n_experts: int, top_k: int,
    capacity_factor: float, dropout: float
) -> MoEMLP:
    """根据一个已有的 MLP 模块创建并初始化一个 MoEMLP。"""
    d_model, d_hidden = -1, -1
    # 优先从 fc1, fc2 属性推断维度
    if hasattr(mlp_module, "fc1") and hasattr(mlp_module, "fc2") and \
       isinstance(mlp_module.fc1, nn.Linear) and isinstance(mlp_module.fc2, nn.Linear):
        d_model = mlp_module.fc1.in_features
        d_hidden = mlp_module.fc1.out_features
    else:
        # 备用方案：从子模块中搜索两个线性层
        linears = [m for m in mlp_module.modules() if isinstance(m, nn.Linear)]
        if len(linears) >= 2:
            d_model = linears[0].in_features
            d_hidden = linears[0].out_features
        else:
            raise AssertionError("无法自动推断MoE维度；请传递自定义的构建器。")

    moe = MoEMLP(
        d_model=d_model, d_hidden=d_hidden, n_experts=n_experts,
        top_k=top_k, capacity_factor=capacity_factor, dropout=dropout,
    )
    # 尝试将原 MLP 的权重迁移到第 0 号专家，便于热启动
    if hasattr(mlp_module, "fc1") and hasattr(mlp_module, "fc2"):
        try:
            moe.experts[0].fc1.load_state_dict(mlp_module.fc1.state_dict())
            moe.experts[0].fc2.load_state_dict(mlp_module.fc2.state_dict())
        except Exception as e:
            print(f"[MoE] Warning: Failed to copy weights from original MLP to expert 0. Reason: {e}")
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
    递归地将模型中的 MLP/FFN 子模块替换为 MoEMLP。

    返回: (int) 被成功替换的层的数量。
    """
    candidates = set()
    for full_name, module in root.named_modules():
        if _looks_like_mlp(module) or any(kw in full_name.lower() for kw in name_keywords):
            if isinstance(module, MoEMLP):
                continue
            # 为了避免替换子模块，我们只选择没有父模块也是候选者的模块
            is_top_level_candidate = True
            for parent_name in candidates:
                if full_name.startswith(parent_name + "."):
                    is_top_level_candidate = False
                    break
            if is_top_level_candidate:
                candidates.add(full_name)

    if not candidates:
        print("[MoE] No MLP/FFN candidates found to replace.")
        return 0
    
    # 排序以保证 last_k 的确定性
    sorted_candidates = sorted(list(candidates))

    if select == "all":
        targets = sorted_candidates
    elif select == "last_k":
        targets = sorted_candidates[-k:] if k > 0 else []
    else:
        raise ValueError(f"Unknown select strategy: {select}")

    # 执行替换
    replaced_count = 0
    print(f"[MoE] Found {len(targets)} layer(s) to replace:")
    for full_name in targets:
        try:
            parent, attr = _resolve_parent_and_attr(root, full_name)
            old_module = getattr(parent, attr)
            moe_module = _make_moe_from_mlp(old_module, n_experts, top_k, capacity_factor, dropout)
            setattr(parent, attr, moe_module)
            print(f"   - Replaced '{full_name}' with MoEMLP.")
            replaced_count += 1
        except Exception as e:
            print(f"   - Failed to replace '{full_name}'. Reason: {e}")

    return replaced_count


# --- PyTorch Lightning 集成工具 ---

def collect_moe_aux_loss(root: nn.Module) -> torch.Tensor:
    """从模型的所有 MoEMLP 子模块中收集辅助损失并求和。"""
    aux_loss = None
    for m in root.modules():
        if isinstance(m, MoEMLP) and m._last_aux_loss is not None and m._last_aux_loss != 0:
            if aux_loss is None:
                aux_loss = m._last_aux_loss
            else:
                aux_loss += m._last_aux_loss
    
    if aux_loss is None:
        # 确保总是在模型的设备上返回一个张量
        return torch.tensor(0.0, device=next(root.parameters()).device, dtype=torch.float32)
    return aux_loss

def reset_moe_aux_loss(root: nn.Module):
    """重置所有 MoEMLP 子模块中缓存的辅助损失。"""
    for m in root.modules():
        if isinstance(m, MoEMLP):
            m._last_aux_loss = None

def patch_training_step_with_moe(lightning_module: "pl.LightningModule", aux_coef: float = 1e-2):
    """
    通过 Monkey Patching 的方式修改 LightningModule 的 training_step，
    自动将 MoE 的辅助损失添加到总损失中。
    """
    if not hasattr(lightning_module, "training_step"):
        raise AttributeError("给定的模块没有 training_step 方法可以修补。")

    original_step = lightning_module.training_step

    def patched_step(*args, **kwargs):
        # 执行原始的 training_step
        result = original_step(*args, **kwargs)
        
        # 收集并加权 MoE 辅助损失
        aux_loss = collect_moe_aux_loss(lightning_module) * aux_coef
        reset_moe_aux_loss(lightning_module) # 重置以备下一轮

        # 将辅助损失添加到主损失中
        if isinstance(result, torch.Tensor):
            # training_step 直接返回 loss 张量
            return result + aux_loss
        elif isinstance(result, dict) and "loss" in result:
            # training_step 返回包含 loss 的字典
            result["loss"] = result["loss"] + aux_loss
            # 同时记录辅助损失的值，便于在日志中查看
            lightning_module.log("moe_aux_loss", aux_loss.detach(), on_step=True, on_epoch=False)
            return result
        else:
            print("[MoE Patch] Warning: Could not add aux_loss to training_step result. "
                  "Expected a Tensor or a dict with a 'loss' key.")
            return result
    
    # 替换原始方法
    lightning_module.training_step = patched_step
    print(f"[MoE] Patched `training_step` with aux_coef={aux_coef}.")