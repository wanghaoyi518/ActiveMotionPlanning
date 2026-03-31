# Reachability-Based Planning 中 MPC 求解设计详细分析

## 1. 总体架构

新实现的规划方法采用 **采样式安全 MPC（Sampling-Based Safe MPC）** 方法，核心思想是：
- **采样候选控制序列** → **仿真轨迹** → **检查安全约束** → **选择最优可行解**

与传统的基于优化的 MPC 不同，这里使用**枚举采样**的方式，避免了复杂的非线性优化问题。

---

## 2. MPC 求解流程（`solve` 方法）

### 2.1 完整流程（第416-468行）

```python
def solve(self, state, ilq_results_A, ilq_results_D, theta_prob, beta_distr, active, beta_w):
```

**五个主要步骤：**

1. **Step 1: 采样内部状态**（第449-451行）
   ```python
   self.sampler = InternalStateSampler(beta_distr, theta_prob)
   internal_samples = self.sampler.sample(self.K_int)
   ```
   - 从当前 belief 中采样 `K_int` 个内部状态 `(psi, beta)`
   - 用于后续的人类轨迹仿真

2. **Step 2: 采样机器人候选控制序列**（第453-454行）
   ```python
   candidate_U_R = self.sample_robot_control_sequences(self.K_R, self.N)
   ```
   - 生成 `K_R` 个长度为 `N` 的控制序列
   - 每个序列：`(N, 2)` - `[加速度, 转向角]`

3. **Step 3: 仿真人类轨迹并构建可达集**（第456-463行）
   ```python
   X_H_samples = self.builder.rollout_human_trajectories(...)
   X_H_sets = self.builder.build_reachable_sets(X_H_samples)
   ```
   - 基于内部状态样本和机器人控制序列，仿真人类未来轨迹
   - 构建每个时间步的人类可达集（椭球表示）

4. **Step 4: 求解安全 MPC**（第465行）
   ```python
   u_r_t = self.solve_safe_mpc_sampling(state, X_H_sets, candidate_U_R)
   ```
   - 核心求解函数

5. **Step 5: 返回第一控制量**
   - 返回最优控制序列的第一个控制量 `u_r_t[0]`

---

## 3. 核心 MPC 求解器：`solve_safe_mpc_sampling`

### 3.1 方法签名（第307-324行）

```python
def solve_safe_mpc_sampling(self, x0, X_H_sets, candidate_U_R):
    """
    输入:
        x0: 当前联合状态 [xH, yH, psiH, vH, xR, yR, psiR, vR]
        X_H_sets: 人类可达集列表（每个时间步一个椭球）
        candidate_U_R: 候选机器人控制序列列表
    输出:
        u_r_t: 当前时刻的最优控制 [a, delta]
    """
```

### 3.2 算法设计：两阶段筛选策略

#### **阶段一：可行性筛选 + 代价优化**（第325-376行）

```python
feasible_sequences = []
feasible_costs = []
feasible_distances = []

for U_R_i in candidate_U_R:
    # 1. 仿真机器人轨迹
    xR_traj = simulate_robot_trajectory(U_R_i)
    
    # 2. 检查安全约束
    is_safe, min_dist, violation_count = check_safety_constraints(xR_traj, X_H_sets)
    
    # 3. 检查状态约束
    state_violation = check_state_constraints(xR_traj)
    
    # 4. 如果可行，计算代价并保存
    if is_safe and not state_violation:
        cost = compute_robot_cost(xR_traj, U_R_i)
        feasible_sequences.append(U_R_i)
        feasible_costs.append(cost)
        feasible_distances.append(min_dist)
```

**关键设计点：**

1. **轨迹仿真**（第334-352行）
   - 使用**自行车模型**（Bicycle Model）进行前向仿真
   - 状态更新公式：
     ```python
     x_next = x + v * cos(psi) * dt
     y_next = y + v * sin(psi) * dt
     psi_next = psi + v * tan(delta) * dt / L
     v_next = v + a * dt
     ```
   - 注意：这里使用的是**简化版**自行车模型，与 `VehicleDyanmics.update()` 一致

2. **安全约束检查**（第354-355行）
   - 调用 `check_safety_constraints()` 方法
   - 检查机器人轨迹与人类可达集的距离是否满足 `d_safe`

3. **状态约束检查**（第357-363行）
   - 检查道路边界：`y ∈ [0, rd_width]`
   - 检查速度限制：`v ∈ [0, v_max]`
   - 如果违反任何约束，标记为不可行

4. **代价计算**（第366-367行）
   - 只对**可行序列**计算代价
   - 避免在不可行解上浪费计算

#### **阶段二：最优解选择**（第372-376行）

```python
if len(feasible_sequences) > 0:
    # 有可行解：选择代价最小的
    best_idx = np.argmin(feasible_costs)
    best_sequence = feasible_sequences[best_idx]
else:
    # 无可行解：使用 Fallback 策略
    best_sequence = fallback_strategy()
```

**设计特点：**
- **优先保证安全性**：只从可行解中选择
- **代价最小化**：在可行解中选择代价最小的序列
- **Fallback 机制**：如果没有可行解，使用备选策略

---

### 3.3 Fallback 策略（第377-411行）

当**没有可行序列**时，采用**最小违反策略**：

```python
# 选择违反约束最少的序列
min_violations = np.inf
for U_R_i in candidate_U_R:
    xR_traj = simulate_robot_trajectory(U_R_i)
    _, _, violation_count = check_safety_constraints(xR_traj, X_H_sets)
    
    if violation_count < min_violations:
        min_violations = violation_count
        best_sequence = U_R_i

# 如果仍然没有找到，使用零控制
if best_sequence is None:
    best_sequence = np.zeros((self.N, 2))
```

**设计理念：**
- 即使没有完全可行的解，也要选择一个"相对最好"的解
- 避免机器人完全停止（零控制是最后的选择）

---

## 4. 关键辅助函数

### 4.1 `check_safety_constraints`（第265-305行）

**功能：**检查机器人轨迹是否满足安全约束

**实现逻辑：**
```python
for k in range(self.N):
    xR_pos = xR_traj[k, :2]  # 机器人位置 [x, y]
    X_H_k = X_H_sets[k]      # 人类可达集（椭球）
    
    # 计算到椭球的距离
    dist = distance_point_to_ellipsoid(xR_pos, X_H_k)
    
    # 检查是否违反安全距离
    if dist < self.d_safe:
        violation_count += 1
    
    min_distance = min(min_distance, dist)

is_safe = (violation_count == 0) and (min_distance >= self.d_safe)
```

**关键点：**
- 使用**椭球距离**（马氏距离）而不是欧氏距离
- 返回三个指标：`is_safe`, `min_distance`, `violation_count`
- 允许部分时间步违反约束（通过 `violation_count` 记录）

### 4.2 `distance_point_to_ellipsoid`（第11-54行）

**功能：**计算点到椭球边界的距离

**数学原理：**
- 椭球定义：`(x - μ)ᵀ Σ⁻¹ (x - μ) ≤ 1`
- 马氏距离：`d_mahal = √[(x - μ)ᵀ Σ⁻¹ (x - μ)]`
- 距离计算：
  - 如果 `d_mahal < 1`：点在椭球内部，返回**负距离**
  - 如果 `d_mahal ≥ 1`：点在椭球外部，返回**正距离**

**实现细节：**
```python
# 计算马氏距离
mahal_dist = sqrt(centered @ cov_inv @ centered)

if mahal_dist < 1:
    # 点在内部：返回负距离
    return -(1 - mahal_dist) * sqrt(max_eigenval)
else:
    # 点在外部：返回正距离
    return (mahal_dist - 1) * sqrt(max_eigenval)
```

### 4.3 `compute_robot_cost`（第233-263行）

**功能：**计算机器人轨迹的代价

**代价函数组成：**

1. **参考轨迹跟踪代价**：
   ```python
   cost += Qref * (y - center_line)²      # 中心线偏差
   cost += Qpsi * (psi - 0)²              # 航向偏差
   cost += Qvel * (v - v_desired)²         # 速度偏差
   ```

2. **控制惩罚**：
   ```python
   cost += Racc * a² + Rdel * delta²      # 控制输入惩罚
   ```

**设计特点：**
- **不包含碰撞代价**：因为碰撞通过硬约束（安全距离）处理
- **只考虑机器人自身状态**：不直接考虑人类状态
- **累积代价**：对预测时域内所有时间步求和

### 4.4 `sample_robot_control_sequences`（第206-231行）

**功能：**采样机器人候选控制序列

**实现方式：**
```python
for _ in range(K_R):
    u_seq = np.random.uniform(
        low=self.uR_min,   # [-uR_acc_max, -uR_delta_max]
        high=self.uR_max,  # [uR_acc_max, uR_delta_max]
        size=(N, 2)
    )
    candidate_U_R.append(u_seq)
```

**设计特点：**
- **均匀随机采样**：在控制空间内均匀采样
- **简单高效**：不需要复杂的采样策略（如重要性采样）
- **可扩展**：可以替换为更智能的采样策略（如基于高斯过程、基于历史最优解等）

---

## 5. 设计优势与局限性

### 5.1 优势

1. **简单直观**：
   - 不需要复杂的优化求解器
   - 逻辑清晰，易于理解和调试

2. **保证安全性**：
   - 硬约束：安全距离必须满足
   - Fallback 机制：即使没有完全可行解，也能选择相对安全的解

3. **计算效率**：
   - 并行化友好：每个候选序列可以独立评估
   - 早期剪枝：不可行序列立即丢弃，不计算代价

4. **鲁棒性**：
   - 不依赖优化器的收敛性
   - 即使采样质量不高，也能找到可行解（如果存在）

### 5.2 局限性

1. **采样效率**：
   - 均匀随机采样可能效率较低
   - 需要大量采样（`K_R = 1000`）才能覆盖控制空间

2. **局部最优**：
   - 只能找到采样到的序列中的最优解
   - 可能错过更好的解（如果不在采样集中）

3. **计算复杂度**：
   - 时间复杂度：`O(K_R * N * K_int)`
   - 需要评估 `K_R` 个序列，每个序列需要 `N` 步仿真

4. **轨迹仿真简化**：
   - 使用简化的自行车模型，可能与实际动力学有差异
   - 没有考虑控制输入的平滑性约束

---

## 6. 与原始 MPPI 方法的对比

| 特性 | Reachability-Based MPC | MPPI (原始方法) |
|------|----------------------|----------------|
| **优化方式** | 枚举采样 + 硬约束 | 路径积分 + 软约束 |
| **安全性保证** | 硬约束（安全距离） | 软约束（代价函数） |
| **计算复杂度** | O(K_R × N × K_int) | O(K × K2 × N) |
| **最优性** | 局部最优（采样范围内） | 加权平均（可能更平滑） |
| **实现复杂度** | 简单 | 中等 |
| **可解释性** | 高（明确的安全约束） | 中（权重需要调参） |

---

## 7. 改进建议

### 7.1 采样策略改进

1. **重要性采样**：
   - 基于上一时刻的最优解，在其周围采样
   - 使用高斯分布而非均匀分布

2. **分层采样**：
   - 先粗采样找到可行区域
   - 再在可行区域内精细采样

### 7.2 轨迹仿真改进

1. **使用完整动力学模型**：
   - 替换为 `VehicleDyanmics.update()` 或 `InteractionDynamics.integrate()`
   - 确保与实际仿真一致

2. **考虑控制平滑性**：
   - 添加控制变化率约束
   - 避免控制输入剧烈变化

### 7.3 约束处理改进

1. **软约束**：
   - 对于轻微违反约束的情况，使用惩罚函数而非硬拒绝
   - 平衡安全性和可行性

2. **多目标优化**：
   - 同时考虑代价和安全性
   - 使用 Pareto 最优解

---

## 8. 代码执行流程示例

假设 `K_R = 1000`, `N = 8`, `K_int = 50`：

```
1. 采样 50 个内部状态 (psi, beta)
2. 采样 1000 个机器人控制序列（每个长度 8）
3. 对每个控制序列：
   a. 仿真 50 条人类轨迹（基于 50 个内部状态）
   b. 构建 8 个时间步的人类可达集（椭球）
4. 对每个机器人控制序列：
   a. 仿真机器人轨迹（8 步）
   b. 检查安全约束（与 8 个椭球的距离）
   c. 检查状态约束（道路边界、速度）
   d. 如果可行，计算代价
5. 从可行序列中选择代价最小的
6. 返回第一个控制量
```

**总计算量：**
- 人类轨迹仿真：`1000 × 50 × 8 = 400,000` 次动力学更新
- 机器人轨迹仿真：`1000 × 8 = 8,000` 次动力学更新
- 安全约束检查：`1000 × 8 = 8,000` 次距离计算

---

## 9. 总结

新实现的 MPC 求解器采用**采样式枚举 + 硬约束**的设计理念：

1. **核心思想**：通过大量采样候选控制序列，找到既安全又低代价的解
2. **安全保证**：通过硬约束（安全距离）确保安全性
3. **实现简单**：避免了复杂的非线性优化问题
4. **可扩展性**：易于改进采样策略和约束处理方式

这种设计在**安全性优先**的场景下特别适用，虽然可能牺牲一些最优性，但能提供更强的安全保证。

