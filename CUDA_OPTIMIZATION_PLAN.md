# Reachability-Based Planning CUDA 并行优化方案

## 1. 当前实现分析

### 1.1 当前瓶颈（NumPy 串行实现）

当前 `reachability_planner.py` 中的主要计算瓶颈：

```python
# 串行循环：O(K_R) 复杂度
for U_R_i in candidate_U_R:  # K_R = 1000
    # 1. 轨迹仿真：O(N)
    for k in range(self.N):  # N = 8
        xR_traj[k+1] = simulate(xR_traj[k], U_R_i[k])
    
    # 2. 安全约束检查：O(N)
    for k in range(self.N):
        dist = distance_point_to_ellipsoid(xR_pos, X_H_sets[k])
    
    # 3. 代价计算：O(N)
    cost = compute_robot_cost(xR_traj, U_R_i)
```

**总复杂度：** `O(K_R × N)`，串行执行

### 1.2 MPPI 的 CUDA 并行化策略

MPPI 使用 PyTorch 进行批量并行：

```python
# 批量采样：一次性生成 K 个序列
self.ur_sampled_action = torch.distributions.Uniform(...).sample((K, N))

# 批量轨迹仿真：向量化操作
xR = self.xR_0.view(1, -1).repeat(K, 1)  # (K, 4)
for t in range(N):
    uR = sampled_actions[:, t]  # (K, 2)
    xR = self._dynamics(xR, uR)  # 批量更新 (K, 4)

# 批量代价计算：向量化
cost = self.Qref * ((xR[:,1] - center)**2)  # (K,)
```

**关键特点：**
- 所有张量在 CUDA 设备上：`.to(device='cuda')`
- 批量操作：一次处理 K 个序列
- 向量化计算：避免 Python 循环

---

## 2. CUDA 优化方案设计

### 2.1 可并行化的部分

#### **高优先级（计算密集）：**

1. **机器人轨迹仿真**（`solve_safe_mpc_sampling`）
   - 当前：串行循环 `K_R` 次
   - 优化：批量仿真 `K_R` 个序列

2. **安全约束检查**
   - 当前：串行检查每个时间步
   - 优化：批量计算所有序列的距离

3. **代价计算**
   - 当前：串行计算每个序列
   - 优化：向量化批量计算

#### **中优先级（可优化）：**

4. **人类轨迹仿真**（`reachability_builder.py`）
   - 当前：嵌套循环 `K_R × K_int`
   - 优化：批量仿真所有组合

5. **可达集构建**
   - 当前：逐个时间步构建
   - 优化：批量计算均值和协方差

---

## 3. CUDA 优化实现方案

### 3.1 核心设计：批量处理

将串行循环改为批量张量操作：

```python
# 当前实现（串行）
for U_R_i in candidate_U_R:  # K_R 次循环
    xR_traj = simulate_trajectory(U_R_i)
    cost = compute_cost(xR_traj)

# CUDA 优化（批量）
# 一次性处理所有序列
U_R_all = torch.stack(candidate_U_R)  # (K_R, N, 2)
xR_traj_all = batch_simulate_trajectories(U_R_all)  # (K_R, N+1, 4)
costs_all = batch_compute_costs(xR_traj_all)  # (K_R,)
```

### 3.2 具体实现步骤

#### **Step 1: 修改 `ReachabilityPlanner` 类初始化**

```python
def __init__(self):
    # 添加 CUDA 支持
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.dtype = torch.float64
    
    # 原有属性...
```

#### **Step 2: 批量轨迹仿真函数**

```python
def batch_simulate_robot_trajectories(self, U_R_all, xR_0):
    """
    批量仿真机器人轨迹
    
    参数:
        U_R_all: torch.Tensor, shape (K_R, N, 2)
        xR_0: torch.Tensor, shape (4,)
    
    返回:
        xR_traj_all: torch.Tensor, shape (K_R, N+1, 4)
    """
    K_R, N, _ = U_R_all.shape
    
    # 初始化轨迹张量
    xR_traj = torch.zeros((K_R, N+1, 4), 
                         device=self.device, dtype=self.dtype)
    xR_traj[:, 0] = xR_0.unsqueeze(0).repeat(K_R, 1)
    
    # 批量仿真（向量化）
    for k in range(N):
        uR_k = U_R_all[:, k, :]  # (K_R, 2)
        xR_k = xR_traj[:, k, :]  # (K_R, 4)
        
        # 向量化自行车模型更新
        delta = uR_k[:, 1]  # (K_R,)
        a = uR_k[:, 0]      # (K_R,)
        v = xR_k[:, 3]      # (K_R,)
        psi = xR_k[:, 2]    # (K_R,)
        
        xR_traj[:, k+1, 0] = xR_k[:, 0] + v * torch.cos(psi) * self.dt
        xR_traj[:, k+1, 1] = xR_k[:, 1] + v * torch.sin(psi) * self.dt
        xR_traj[:, k+1, 2] = xR_k[:, 2] + v * torch.tan(delta) * self.dt / self.L
        xR_traj[:, k+1, 3] = xR_k[:, 3] + a * self.dt
    
    return xR_traj
```

#### **Step 3: 批量安全约束检查**

```python
def batch_check_safety_constraints(self, xR_traj_all, X_H_sets):
    """
    批量检查安全约束
    
    参数:
        xR_traj_all: torch.Tensor, shape (K_R, N+1, 4)
        X_H_sets: list of dicts (ellipsoids)
    
    返回:
        is_safe_all: torch.Tensor, shape (K_R,), bool
        min_distances: torch.Tensor, shape (K_R,)
        violation_counts: torch.Tensor, shape (K_R,)
    """
    K_R, N_plus_1, _ = xR_traj_all.shape
    N = N_plus_1 - 1
    
    # 将椭球转换为张量
    means = torch.stack([torch.tensor(X_H_k['mean'], 
                                      device=self.device) 
                         for X_H_k in X_H_sets])  # (N, 2)
    covs = torch.stack([torch.tensor(X_H_k['cov'], 
                                     device=self.device) 
                        for X_H_k in X_H_sets])  # (N, 2, 2)
    
    # 批量计算距离
    min_distances = torch.full((K_R,), float('inf'), 
                              device=self.device, dtype=self.dtype)
    violation_counts = torch.zeros((K_R,), 
                                   device=self.device, dtype=torch.long)
    
    for k in range(N):
        xR_pos_k = xR_traj_all[:, k, :2]  # (K_R, 2)
        mean_k = means[k]  # (2,)
        cov_k = covs[k]    # (2, 2)
        
        # 批量计算马氏距离
        centered = xR_pos_k - mean_k.unsqueeze(0)  # (K_R, 2)
        cov_inv = torch.inverse(cov_k)  # (2, 2)
        
        # 批量马氏距离计算
        mahal_dist_sq = torch.sum(centered @ cov_inv * centered, dim=1)  # (K_R,)
        mahal_dist = torch.sqrt(mahal_dist_sq)  # (K_R,)
        
        # 计算到椭球边界的距离
        eigenvals = torch.linalg.eigvals(cov_k).real
        max_eigenval = torch.max(eigenvals)
        
        # 批量距离计算
        dist_k = torch.where(
            mahal_dist < 1.0,
            -(1.0 - mahal_dist) * torch.sqrt(max_eigenval),
            (mahal_dist - 1.0) * torch.sqrt(max_eigenval)
        )  # (K_R,)
        
        # 更新最小距离
        min_distances = torch.minimum(min_distances, dist_k)
        
        # 统计违反次数
        violation_counts += (dist_k < self.d_safe).long()
    
    # 判断是否安全
    is_safe_all = (violation_counts == 0) & (min_distances >= self.d_safe)
    
    return is_safe_all, min_distances, violation_counts
```

#### **Step 4: 批量代价计算**

```python
def batch_compute_robot_costs(self, xR_traj_all, U_R_all):
    """
    批量计算机器人轨迹代价
    
    参数:
        xR_traj_all: torch.Tensor, shape (K_R, N+1, 4)
        U_R_all: torch.Tensor, shape (K_R, N, 2)
    
    返回:
        costs: torch.Tensor, shape (K_R,)
    """
    K_R, N_plus_1, _ = xR_traj_all.shape
    N = N_plus_1 - 1
    
    # 初始化代价张量
    costs = torch.zeros((K_R,), device=self.device, dtype=self.dtype)
    
    # 批量计算每个时间步的代价
    for k in range(N):
        xR_k = xR_traj_all[:, k, :]  # (K_R, 4)
        uR_k = U_R_all[:, k, :]      # (K_R, 2)
        
        # 向量化代价计算
        costs += self.Qref * ((xR_k[:, 1] - self.xlim[1] / 2)**2)
        costs += self.Qpsi * ((xR_k[:, 2] - 0.0)**2)
        costs += self.Qvel * ((xR_k[:, 3] - self.xlim[3])**2)
        costs += self.Racc * (uR_k[:, 0]**2) + self.Rdel * (uR_k[:, 1]**2)
    
    return costs
```

#### **Step 5: 批量状态约束检查**

```python
def batch_check_state_constraints(self, xR_traj_all):
    """
    批量检查状态约束
    
    返回:
        state_violations: torch.Tensor, shape (K_R,), bool
    """
    # 检查道路边界和速度限制
    y_violation = (xR_traj_all[:, :, 1] < 0.0) | \
                  (xR_traj_all[:, :, 1] > self.xlim[1])
    v_violation = (xR_traj_all[:, :, 3] < 0.0) | \
                  (xR_traj_all[:, :, 3] > self.xlim[7])
    
    # 任何时间步违反都标记为不可行
    state_violations = torch.any(y_violation | v_violation, dim=1)
    
    return state_violations
```

#### **Step 6: 重构 `solve_safe_mpc_sampling`**

```python
def solve_safe_mpc_sampling(self, x0, X_H_sets, candidate_U_R):
    """
    CUDA 优化版本的安全 MPC 求解
    """
    # 转换为 PyTorch 张量
    xR_0 = torch.tensor(x0[4:8], device=self.device, dtype=self.dtype)
    
    # 批量处理：将所有候选序列转换为张量
    U_R_all = torch.stack([
        torch.tensor(U_R_i, device=self.device, dtype=self.dtype) 
        for U_R_i in candidate_U_R
    ])  # (K_R, N, 2)
    
    # Step 1: 批量仿真机器人轨迹
    xR_traj_all = self.batch_simulate_robot_trajectories(U_R_all, xR_0)
    
    # Step 2: 批量检查安全约束
    is_safe_all, min_distances, violation_counts = \
        self.batch_check_safety_constraints(xR_traj_all, X_H_sets)
    
    # Step 3: 批量检查状态约束
    state_violations = self.batch_check_state_constraints(xR_traj_all)
    
    # Step 4: 筛选可行序列
    feasible_mask = is_safe_all & (~state_violations)
    
    if torch.any(feasible_mask):
        # 有可行解：计算代价并选择最优
        costs_all = self.batch_compute_robot_costs(xR_traj_all, U_R_all)
        
        # 只考虑可行序列的代价
        feasible_costs = torch.where(
            feasible_mask,
            costs_all,
            torch.tensor(float('inf'), device=self.device)
        )
        
        best_idx = torch.argmin(feasible_costs).item()
        best_sequence = U_R_all[best_idx]
    else:
        # 无可行解：选择违反最少的
        total_violations = violation_counts + state_violations.long()
        best_idx = torch.argmin(total_violations).item()
        best_sequence = U_R_all[best_idx]
    
    # 返回第一个控制量（转换为 numpy）
    return best_sequence[0].cpu().numpy()
```

---

## 4. 性能提升预估

### 4.1 理论加速比

假设 `K_R = 1000`, `N = 8`：

| 操作 | 串行时间 | CUDA 批量时间 | 加速比 |
|------|---------|--------------|--------|
| 轨迹仿真 | 1000 × 8 = 8000 次 | 1 次批量（并行） | ~100-500x |
| 安全约束检查 | 1000 × 8 = 8000 次 | 1 次批量（并行） | ~100-500x |
| 代价计算 | 1000 × 8 = 8000 次 | 1 次批量（并行） | ~100-500x |

**总体加速比预估：** 50-200x（取决于 GPU 性能）

### 4.2 内存需求

```python
# 主要张量大小
U_R_all: (K_R, N, 2) = (1000, 8, 2) = 16KB (float64)
xR_traj_all: (K_R, N+1, 4) = (1000, 9, 4) = 288KB
costs_all: (K_R,) = 8KB
is_safe_all: (K_R,) = 1KB

# 总内存需求：~500KB（非常小，GPU 完全能承受）
```

---

## 5. 实现注意事项

### 5.1 数据转换开销

- **CPU ↔ GPU 数据传输**：尽量减少转换次数
- **建议**：在 GPU 上保持数据，只在最后返回结果时转换

### 5.2 混合精度

- 可以考虑使用 `float32` 而非 `float64` 以提升性能
- 但需要验证数值精度是否足够

### 5.3 批处理大小

- 如果 `K_R` 太大，可以分批处理
- 例如：每次处理 1000 个，分多次处理

---

## 6. 实现优先级

### **Phase 1: 核心优化（最大收益）**
1. ✅ 批量机器人轨迹仿真
2. ✅ 批量安全约束检查
3. ✅ 批量代价计算
4. ✅ 批量状态约束检查

**预期加速：** 50-100x

### **Phase 2: 扩展优化（进一步加速）**
5. 批量人类轨迹仿真（`reachability_builder.py`）
6. 批量可达集构建

**预期加速：** 额外 2-5x

---

## 7. 代码结构建议

### 7.1 保持兼容性

```python
class ReachabilityPlanner:
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.device = "cuda"
            # CUDA 优化版本
        else:
            self.device = "cpu"
            # NumPy 版本（当前实现）
```

### 7.2 渐进式迁移

1. **第一步**：实现 CUDA 版本，保留 NumPy 版本作为 fallback
2. **第二步**：测试和验证 CUDA 版本的正确性
3. **第三步**：性能对比和调优
4. **第四步**：如果性能提升显著，可以移除 NumPy 版本

---

## 8. 预期效果

### 8.1 计算时间对比

| 方法 | 当前（NumPy） | CUDA 优化 | 加速比 |
|------|--------------|-----------|--------|
| `solve_safe_mpc_sampling` | ~100-500ms | ~1-5ms | 50-200x |
| 总规划时间 | ~200-1000ms | ~5-20ms | 20-100x |

### 8.2 适用场景

- **实时性要求高**：CUDA 版本更适合
- **CPU 资源受限**：CUDA 版本可以释放 CPU
- **大规模采样**：`K_R` 越大，CUDA 优势越明显

---

## 9. 总结

新规划方法的 MPC 求解**完全可以**用 CUDA 并行优化，而且：

1. **并行化潜力大**：主要计算都是批量操作，天然适合 GPU
2. **实现相对简单**：主要是将 NumPy 循环改为 PyTorch 批量操作
3. **性能提升显著**：预期 50-200x 加速
4. **内存需求小**：即使 `K_R=1000`，内存需求也很小

**建议实施步骤：**
1. 先实现核心的批量轨迹仿真和约束检查
2. 验证正确性
3. 性能测试和调优
4. 扩展到人类轨迹仿真的批量处理

