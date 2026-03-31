# Active Inference 规划层改造为 Reachability-Based 方法的代码实现指南

本文档描述如何在现有的 Active Inference 代码基础上，将规划层从“路径积分加权 (MPPI-style)”改造为“基于可达集 (reachability) 的安全规划”，同时保持 **Problem Formulation 与 belief 更新模块不变**。你可以将此文档视为对原论文《Active Inference-Based Planning for Safe Human-Robot Interaction》中 Algorithm 1 的“Reachability 版本”实现说明。

---

## 1. 总体改动概览

### 1.1 不变的部分（可以原封不动复用）

以下模块和数据结构可以保持不变：

1. **问题建模层：**

- 状态：\\(x_t\\)
- 控制：\\(u_t^r\\)（机器人）、\\(u_t^h\\)（人类）
- 动力学：
  \\[
  x_{t+1} = f(x_t) + B_r(x_t) u_t^r + B_h(x_t) u_t^h
  \\]
- 人类动作模型：
  \\[
  P(u^h \\mid x, u^r, \\psi, \\beta)
  \\]
  该模型由 IRL + 理性系数 \\(\\beta\\) 的噪声理性假设得到，一般为 soft-optimal / 高斯形式。

2. **内部状态 belief 结构：**

- 人类类型：\\(\\psi\\)（有限离散集合，例如 \\(\\{\\text{attentive}, \\text{distracted}\\}\\)）
- 理性系数：\\(\\beta\\)（连续实数，带截断正态先验）
- 当前时刻的联合 belief：
  \\[
  b_t(\\psi, \\beta) = P(\\psi, \\beta \\mid I_t)
  \\]
  代码中可以表示为：
  ```python
  belief = {
      "psi_probs": {psi1: p1, psi2: p2, ...},   # P(psi | I_t)
      "beta_dists": {
          psi1: TruncatedNormal(mu_beta[psi1], Sigma_beta[psi1], beta_low, beta_high),
          ...
      }
  }
  ```

3. **belief 更新模块（IV-A）：**

这一块代码 **完全不动**，包括：

- 根据当前观测 \\(u_t^h\\)、上一时刻状态 \\(x_{t-1}\\) 与控制 \\(u_{t-1}^r\\)，用贝叶斯公式更新 \\(P(\\beta \\mid I_t, \\psi)\\)：  
  - 用似然 \\(P(u_t^h \\mid x_{t-1}, u_{t-1}^r, \\psi, \\beta)\\)，  
  - 结合上一时刻先验 \\(P(\\beta \\mid I_{t-1}, \\psi)\\)，  
  - 用拉普拉斯近似得到新的均值/协方差，再做区间截断。
- 使用 \\(P(u_t^h \\mid x_{t-1}, u_{t-1}^r, \\psi)\\) 和旧的 \\(P(\\psi \\mid I_{t-1})\\) 更新 \\(P(\\psi \\mid I_t)\\) 的离散贝叶斯步骤。

在代码层面，这通常对应如下函数：

```python
belief = update_belief(belief, x_prev, u_r_prev, u_h_t)
# 内部调用:
# - update_beta_posterior(...)
# - update_psi_posterior(...)
```

4. **基础工具函数：**

可直接复用：

- `step_dynamics(x, u_r, u_h)`：动力学仿真 \\(x_{t+1} = f(x_t)+B_r u_r + B_h u_h\\)
- `sample_human_action(x, u_r, psi, beta)`：从 \\(P(u^h \\mid x,u^r,\\psi,\\beta)\\) 采样
- 单步代价函数 `l_r(x, u_r, u_h)`（可用于新的 MPC 目标）

---

### 1.2 需要删除 / 停用的部分（原规划层）

Active Inference 原版 Algorithm 1 中，用于 **路径积分式规划** 的部分需要被停用或不再调用，包括：

1. **机器人动作序列采样 + 权重计算相关逻辑：**

- 采样机器人动作序列的函数（本身可以重用，但用途改变）：
  ```python
  sample_robot_action_sequences(K1, N)
  ```
- 在每条轨迹 (i,j) 上进行 **未来 belief 虚拟更新** 的代码（即：沿预测时域假想观测 \\(u_k^h\\) 并更新 \\(b_k^{i,j}\\) 的那部分）。
- 轨迹代价与信息量求值：
  - `compute_trajectory_cost(J_r[i,j])`
  - `compute_information_measure(b_k^{i,j} -> b^{i,j})`
- 根据 \\(J^{i,j}, b^{i,j}\\) 计算权重并加权求平均：
  - `compute_weights(w_i)`
  - `u_r_star = sum_i w_i * u_r^i / sum_i w_i`

2. **原规划函数 `plan_control_active_inference` 的内部实现：**

原函数大致结构类似：

```python
def plan_control_active_inference(x_t, belief, params):
    # 1. 采样机器人动作序列
    # 2. 对每条轨迹采样人类动作 & 更新未来 belief
    # 3. 计算轨迹代价 J^{i,j} 和信息指标 b^{i,j}
    # 4. 计算权重 w_i
    # 5. 加权平均得到 u_r*
    return u_r_star[0]  # 第一控制量
```

**我们会完全替换其内部逻辑**，但外部接口（输入 x_t, belief, params，输出 u_r_t）可以保持一致，以减小对主循环的侵入性。

---

### 1.3 需要新增 / 替换的模块（新规划层）

目标是实现一个新的规划函数：

```python
def plan_control_reachability(x_t, belief, params):
    ...
    return u_r_t
```

其核心包含三类新模块：

1. **内部状态采样（从 belief 采样 (ψ, β)）：**

```python
def sample_internal_states(belief, K_int):
    """
    从当前 belief 中采样 K_int 个内部状态样本 (psi, beta)
    belief: {
        "psi_probs": {psi: prob, ...},
        "beta_dists": {psi: TruncatedNormal(...), ...}
    }
    返回: [(psi_s, beta_s) for s in range(K_int)]
    """
```

2. **人类未来轨迹采样 & 可达集合构造：**

```python
def rollout_human_trajectories(x_t, internal_samples, candidate_U_R, N):
    """
    基于内部状态样本和机器人候选控制序列，仿真人类未来轨迹。
    返回每个时间步的人类状态样本集合 X_H_samples[k]。
    """
```

```python
def build_reachable_sets(X_H_samples):
    """
    从每个时间步的人类位置样本集合 X_H_samples[k] 构造凸包，得到可达集合 X_H^k。
    """
```

3. **带可达集合约束的安全 MPC：**

```python
def solve_safe_mpc_with_reachability(x_t, X_H_sets, params):
    """
    在约束 dist(x_R^k, X_H^k) >= d_safe 下，
    求解机器人在时域 N 内的最优控制序列 U_R_star。
    """
```

---

## 2. 新规划函数 `plan_control_reachability` 详细伪代码

下面给出接近代码风格的伪代码，实现 “工作流 B”，但使用的符号和 belief 完全遵循 Active Inference 论文。

```python
def plan_control_reachability(x_t, belief, params):
    """
    输入:
        x_t     : 当前状态
        belief  : 当前内部状态联合分布 b_t(psi, beta)
                  belief = {
                      "psi_probs": {psi: prob},
                      "beta_dists": {psi: TruncatedNormal(...)}
                  }
        params  : 包含 N, d_safe, K_int, K_R 等超参数
    输出:
        u_r_t   : 当前时刻的机器人控制输入 u_t^r
    """
    N      = params.N_horizon      # 预测时域长度
    K_int  = params.K_internal     # 内部状态采样数量
    K_R    = params.K_robot_seq    # 机器人候选控制序列数量
    d_safe = params.d_safe         # 安全距离

    # ---------- 1. 从 belief 中采样内部状态 (psi, beta) ----------
    internal_samples = sample_internal_states(belief, K_int)
    # internal_samples = [ (psi_s, beta_s) for s in range(K_int) ]

    # ---------- 2. 构造机器人候选控制序列 ----------
    # 可以重用原先的 sample_robot_action_sequences，只是用途不同
    candidate_U_R = sample_robot_action_sequences(K_R, N, params)
    # candidate_U_R[i] = [u_r^{i,0}, ..., u_r^{i,N-1}]

    # ---------- 3. 基于内部状态样本和候选 U_R 仿真人类轨迹 ----------
    # X_H_samples[k] 存储预测第 k 步的人类位置样本
    X_H_samples = [ [] for _ in range(N) ]

    for i in range(K_R):  # 遍历机器人候选控制序列
        U_R_i = candidate_U_R[i]

        for (psi_s, beta_s) in internal_samples:

            x_pred = x_t.copy()  # 预测起点

            for k in range(N):
                u_r_k = U_R_i[k]

                # 人类动作采样: P(u^h | x_pred, u_r_k, psi_s, beta_s)
                u_h_k = sample_human_action(x_pred, u_r_k, psi_s, beta_s)

                # 人机动力学推进
                x_pred = step_dynamics(x_pred, u_r_k, u_h_k)

                # 提取人类部分的状态 (例如车辆位置)
                x_h_k = extract_human_state(x_pred)
                X_H_samples[k].append(x_h_k)

    # ---------- 4. 对每个时间步构造人类可达集合 X_H^k ----------
    X_H_sets = []

    for k in range(N):
        # 使用几何库构造凸包作为可达集外包
        hull_k = convex_hull(X_H_samples[k])  # 例如返回顶点列表、半空间表示等
        X_H_sets.append(hull_k)

    # ---------- 5. 在可达集合约束下解一个安全 MPC ----------
    U_R_star = solve_safe_mpc_with_reachability(x_t, X_H_sets, params)

    # ---------- 6. 输出当前时刻控制 ----------
    u_r_t = U_R_star[0]
    return u_r_t
```

### 关键函数设计说明

#### 2.1 `sample_internal_states(belief, K_int)`

```python
def sample_internal_states(belief, K_int):
    psi_probs = belief["psi_probs"]
    beta_dists = belief["beta_dists"]

    psi_list = list(psi_probs.keys())
    prob_list = [psi_probs[psi] for psi in psi_list]

    samples = []
    for _ in range(K_int):
        # 先按 P(psi | I_t) 采样 psi_s
        psi_s = random_choice(psi_list, p=prob_list)
        # 再从对应的截断正态分布采样 beta_s
        beta_dist = beta_dists[psi_s]
        beta_s = beta_dist.sample()
        samples.append((psi_s, beta_s))

    return samples
```

#### 2.2 `convex_hull(X_H_samples[k])`

这里需要依赖几何库（例如 Python 的 `scipy.spatial.ConvexHull`），伪代码：

```python
from scipy.spatial import ConvexHull
import numpy as np

def convex_hull(points):
    """
    points: list of 2D or 3D points, e.g. [(x,y), ...]
    返回 hull 对象，可包含:
        - hull.vertices: 顶点索引
        - hull.points: 所有点
        - hull.equations: 半空间表示 (法向量与偏移)
    """
    pts = np.array(points)
    hull = ConvexHull(pts)
    return hull
```

之后在 `solve_safe_mpc_with_reachability` 中，需要基于 hull 的表示来实现约束：
\\[
\\mathrm{dist}(x_{R,k}, X_H^k) \\ge d_{\\mathrm{safe}}.
\\]

这可以通过以下方式实现：

- 若凸包用 **半空间交** 表示：
  \\[
  X_H^k = \\{ x \\mid A_k x \\le b_k \\\}
  \\]
  则“距离至少为 \\(d_{safe}\\)”可以转化为一组非线性约束或引入 slack 变量和 big-M 技巧，在 MILP 中实现。

- 简化版实现：保持机器人在某些“安全半平面”外，可以以 conservative 的方式实现。

#### 2.3 `solve_safe_mpc_with_reachability(x_t, X_H_sets, params)`

此函数结构类似于标准 MPC：

```python
def solve_safe_mpc_with_reachability(x_t, X_H_sets, params):
    N      = params.N_horizon
    d_safe = params.d_safe

    # 使用某个优化建模工具 (例如 CasADi, Pyomo, Gurobi, etc.)
    model = create_optimization_model()

    # 决策变量: 机器人状态与控制
    x_R = model.add_state_vars(N+1)  # x_R[0..N]
    u_R = model.add_control_vars(N)  # u_R[0..N-1]

    # 初始条件
    model.add_constraint(x_R[0] == extract_robot_state(x_t))

    # 动力学与控制约束
    for k in range(N):
        model.add_constraint(x_R[k+1] == f_R(x_R[k], u_R[k]))
        model.add_constraint(control_bounds(u_R[k]))

    # 几何安全约束: dist(x_R[k], X_H^k) >= d_safe
    for k in range(N):
        add_distance_constraint_to_hull(model, x_R[k], X_H_sets[k], d_safe)

    # 目标函数: 累积代价
    cost = 0
    for k in range(N):
        cost += l_r_state_only(x_R[k], u_R[k])  # 可用原来的 l_r 或其简化版

    model.set_objective_minimize(cost)

    # 求解
    result = model.solve()
    U_R_star = [ result.value(u_R[k]) for k in range(N) ]
    return U_R_star
```

其中 `add_distance_constraint_to_hull` 的实现依赖你选择的优化器和凸包表示形式，这部分属于几何+优化的工程细节，可按你原 SRP 方案进行实现或移植。

---

## 3. 主循环中的改造

原主循环大概率类似：

```python
while not done:
    x_t = observe_state()
    u_h_t = observe_human_action()

    # belief 更新 (IV-A)
    belief = update_belief(belief, x_prev, u_r_prev, u_h_t)

    # 原始 Active Inference 规划
    u_r_t = plan_control_active_inference(x_t, belief, params)

    apply_control(u_r_t)

    x_prev = x_t
    u_r_prev = u_r_t
    t += 1
```

改造后，只需要 **一处替换**：

```python
while not done:
    x_t = observe_state()
    u_h_t = observe_human_action()

    # belief 更新完全保持不变
    belief = update_belief(belief, x_prev, u_r_prev, u_h_t)

    # 使用新的 Reachability-Based 规划
    u_r_t = plan_control_reachability(x_t, belief, params)

    apply_control(u_r_t)

    x_prev = x_t
    u_r_prev = u_r_t
    t += 1
```

---

## 4. 文件与模块层面的改动建议（Checklist）

### 4.1 保留的文件/模块

- `human_model.py`  
  - IRL 得到的回报函数与人类策略  
  - `sample_human_action(...)`  
- `belief_update.py`  
  - `update_belief(...)`（实现 IV-A，支持 ψ 与 β 的更新）  
- `dynamics.py`  
  - `step_dynamics(...)`  
- `cost_functions.py`  
  - `l_r(...)` 或分解后的 `l_r_state_only(...)`

### 4.2 停用或删除的代码片段

- `compute_path_integral_weights(...)`
- `update_future_belief_along_rollout(...)`（轨迹仿真中虚拟 belief 更新）  
- `plan_control_active_inference(...)` 内部的 “双重采样 + 权重求平均” 逻辑：

  ```python
  for i in range(K1):
      for j in range(K2):
          # rollout x^{i,j}_k, u^{i,j}_h,k
          # 更新虚拟 b^{i,j}_k
          # 计算 J^{i,j}, b^{i,j}
  # 计算 w_i, 加权得到 u_r*
  ```

### 4.3 新增的文件/模块（推荐组织方式）

- `internal_state_sampler.py`
  - `sample_internal_states(belief, K_int)`
- `reachability_builder.py`
  - `rollout_human_trajectories(...)`
  - `build_reachable_sets(...)`
- `safe_mpc_reachability.py`
  - `solve_safe_mpc_with_reachability(x_t, X_H_sets, params)`
- `planner_reachability.py`
  - `plan_control_reachability(x_t, belief, params)`

主循环所在文件，只需将原来的

```python
u_r_t = plan_control_active_inference(x_t, belief, params)
```

替换为

```python
u_r_t = plan_control_reachability(x_t, belief, params)
```

---

## 5. 总结

- **Problem formulation 与 belief 更新**：完全沿用 Active Inference 论文的设定与实现（内部状态 = 类型 ψ + 理性系数 β，IV-A 的贝叶斯更新与拉普拉斯近似）。  
- **规划层改动**：  
  - 不再对机器人动作序列做路径积分加权；  
  - 换成：从 belief 采样内部状态 → 采样人类未来轨迹 → 构造每步的人类可达集合（凸包） → 在几何安全约束 dist(x_R^k, X_H^k) ≥ d_safe 下做 MPC。  
- **代码层改造**：  
  - 保留 belief 更新与人类模型模块；  
  - 替换 `plan_control_active_inference` 的内部逻辑，新增 `plan_control_reachability` 及其依赖模块；  
  - 主循环只需一行函数调用的替换，即可切换到新的 reachability-based 规划框架。

你可以直接在代码仓库中添加本 `.md` 文件（例如 `docs/active_to_reachability_refactor_guide.md`），作为实现改造的开发文档，也可以进一步补充具体函数签名、类结构和你实际使用的优化器 API。
