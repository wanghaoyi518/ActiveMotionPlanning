分析代码结构，制定将 Active Inference 规划替换为基于 sampling-based reachability analysis 的详细方案。


[4 tools called]


基于代码结构，制定将 Active Inference 规划替换为基于 sampling-based reachability analysis 的详细实现方案。

## 详细实现方案：Reachability-Based 规划模块集成

### 一、总体架构设计

#### 1.1 设计原则
- 保持接口一致性：新模块与现有 `mppi` 类使用相同接口
- 模块化：新功能独立文件，便于维护和对比
- 最小侵入：主循环仅需少量修改即可切换
- 可复用：共享动力学、人类模型等基础模块

#### 1.2 目录结构规划

```
motion_planning/
├── __init__.py
├── dynamics.py                    # 保持不变
├── inference.py                   # 保持不变（belief更新）
├── vehicle_model.py               # 保持不变
├── MPPI.py                        # 保持不变（原Active Inference规划）
├── reachability_planner.py       # 【新增】Reachability规划主类
├── internal_state_sampler.py      # 【新增】内部状态采样模块
├── reachability_builder.py        # 【新增】可达集构造模块
└── safe_mpc_solver.py            # 【新增】安全MPC求解器（可选，如果使用优化器）
```

### 二、新增文件详细设计

#### 2.1 `motion_planning/internal_state_sampler.py`

目的：从 belief 中采样内部状态 (ψ, β)

类设计：
- 类名：`InternalStateSampler`
- 方法：
  - `__init__(self, beta_distr, theta_prob)`
    - 参数：`beta_distr` (beta_prob_distr对象), `theta_prob` (list，[P(attentive), P(distracted)])
    - 作用：初始化采样器，保存当前 belief
  - `sample(self, K_int)`
    - 参数：`K_int` (int，采样数量)
    - 返回：`list of tuples`，`[(psi_s, beta_s), ...]`，每个元素为 (人类类型, 理性系数)
    - 逻辑：
      1. 按 `theta_prob` 采样 `psi_s` ('a' 或 'd')
      2. 从对应 `beta_distr.a` 或 `beta_distr.d` 的截断正态分布采样 `beta_s`
      3. 返回 K_int 个样本

依赖关系：
- 使用：`motion_planning/inference.py` 中的 `beta_prob_distr` 类
- 被使用：`reachability_planner.py`

#### 2.2 `motion_planning/reachability_builder.py`

目的：基于内部状态样本和机器人候选控制序列，仿真人类轨迹并构造可达集

类设计：
- 类名：`ReachabilityBuilder`
- 方法：
  - `__init__(self, dynamics_A, dynamics_D, ilq_results_A, ilq_results_D, dt, L, uH_lim, beta_w)`
    - 参数：
      - `dynamics_A/D`：InteractionDynamics对象（attentive/distracted）
      - `ilq_results_A/D`：ilq_results对象
      - `dt`：时间步长
      - `L`：车辆轴距
      - `uH_lim`：人类控制限制
      - `beta_w`：是否考虑理性系数
    - 作用：初始化，保存动力学和人类模型参数
  - `rollout_human_trajectories(self, x0, internal_samples, candidate_U_R, N)`
    - 参数：
      - `x0`：当前状态 (numpy array, shape: [8,])
      - `internal_samples`：内部状态样本列表 `[(psi, beta), ...]`
      - `candidate_U_R`：机器人候选控制序列 `list of arrays`，每个 shape [N, 2]
      - `N`：预测时域长度
    - 返回：`X_H_samples` (list of lists)，`X_H_samples[k]` 包含第 k 步的所有人类位置样本 `[(x, y), ...]`
    - 逻辑：
      1. 遍历每个机器人控制序列 `U_R_i`
      2. 遍历每个内部状态样本 `(psi_s, beta_s)`
      3. 对每个时间步 k：
         - 根据 `psi_s` 选择对应的 iLQ 结果和动力学
         - 计算人类最优动作 `uH_k`（基于 iLQ 反馈控制）
         - 根据 `beta_s` 添加噪声（高斯采样）
         - 使用动力学推进状态
         - 提取人类位置 `(x, y)` 加入 `X_H_samples[k]`
  - `build_reachable_sets(self, X_H_samples, method='convex_hull')`
    - 参数：
      - `X_H_samples`：从 `rollout_human_trajectories` 返回的样本集合
      - `method`：构造方法（'convex_hull' 或 'ellipsoid'）
    - 返回：`X_H_sets` (list)，每个元素是第 k 步的可达集表示
      - 如果 `method='convex_hull'`：返回 `list of ConvexHull objects` (scipy.spatial.ConvexHull)
      - 如果 `method='ellipsoid'`：返回 `list of dict`，包含均值、协方差等
    - 逻辑：
      1. 对每个时间步 k：
         - 收集所有位置样本 `X_H_samples[k]`
         - 使用 scipy.spatial.ConvexHull 构造凸包
         - 或计算椭球参数（均值、协方差）
      2. 返回所有时间步的可达集

依赖关系：
- 使用：
  - `motion_planning/dynamics.py` 中的 `InteractionDynamics`
  - `human_model/solveiLQgame.py` 中的 `ilq_results`
  - `human_model/iLQgame.py` 中的 `get_covariance`
  - `scipy.spatial.ConvexHull`
- 被使用：`reachability_planner.py`

#### 2.3 `motion_planning/reachability_planner.py`

目的：Reachability-Based 规划主类，替代 `mppi` 类

类设计：
- 类名：`ReachabilityPlanner`
- 方法：
  - `__init__(self)`
    - 作用：初始化，设置设备（CPU/GPU），类似 mppi 的初始化
  - `set_params(self, N, K_R, K_int, xlim, ulim, Nx, Nu, dt, L, d_safe)`
    - 参数：
      - `N`：预测时域长度
      - `K_R`：机器人候选控制序列数量
      - `K_int`：内部状态采样数量
      - `xlim, ulim`：状态和控制限制
      - `Nx, Nu`：状态和控制维度
      - `dt, L`：时间步长和轴距
      - `d_safe`：安全距离阈值
    - 作用：设置规划参数
  - `set_cost(self, weight, ...)`
    - 参数：与 `mppi.set_cost()` 相同
    - 作用：设置代价函数权重（用于 MPC 目标函数）
  - `set_human_models(self, dynamics_A, dynamics_D, ilq_results_A, ilq_results_D, beta_w)`
    - 参数：人类模型相关对象
    - 作用：设置人类决策模型
  - `solve(self, state, ilq_results_A, ilq_results_D, theta_prob, beta_distr, active, beta_w)`
    - 参数：与 `mppi.solve_mppi()` 完全相同
    - 返回：`u_r_t` (numpy array, shape: [2,])，当前时刻机器人控制
    - 逻辑（核心算法）：
      1. 创建 `InternalStateSampler` 并采样 `internal_samples`
      2. 采样机器人候选控制序列 `candidate_U_R`（可复用 MPPI 的采样逻辑）
      3. 创建 `ReachabilityBuilder` 并调用 `rollout_human_trajectories()`
      4. 调用 `build_reachable_sets()` 得到 `X_H_sets`
      5. 调用 `solve_safe_mpc()` 或 `solve_safe_mpc_sampling()` 求解最优控制
      6. 返回第一控制量 `u_r_t`
  - `solve_safe_mpc_sampling(self, x0, X_H_sets, candidate_U_R)`
    - 参数：
      - `x0`：当前状态
      - `X_H_sets`：人类可达集列表
      - `candidate_U_R`：候选控制序列
    - 返回：最优控制序列的第一个控制量
    - 逻辑（采样式求解，避免使用优化器）：
      1. 对每个候选控制序列 `U_R_i`：
         - 仿真机器人轨迹 `x_R_traj`
         - 检查每个时间步 k：`dist(x_R_traj[k], X_H_sets[k]) >= d_safe`
         - 如果违反约束，标记为不可行
         - 如果可行，计算代价 `J_i = sum(l_r(x_R[k], u_R[k]))`
      2. 从可行序列中选择代价最小的
      3. 如果没有可行序列，选择违反约束最少的（或使用 fallback 策略）
      4. 返回最优序列的第一个控制量

依赖关系：
- 使用：
  - `internal_state_sampler.py`
  - `reachability_builder.py`
  - `motion_planning/dynamics.py`
  - `motion_planning/inference.py`
- 被使用：`main.py`

#### 2.4 `motion_planning/safe_mpc_solver.py`（可选）

目的：如果使用优化器（如 CasADi, CVXPY），提供基于优化的 MPC 求解器

类设计：
- 类名：`SafeMPCSolver`
- 方法：
  - `solve(self, x0, X_H_sets, params)`
    - 使用优化器求解带可达集约束的 MPC 问题

注意：如果使用采样式求解（推荐，更简单），此文件可选。

### 三、现有文件修改方案

#### 3.1 `main.py` 修改点

位置1：导入部分（第19行附近）
- 当前：`from motion_planning.MPPI import *`
- 修改为：
```python
from motion_planning.MPPI import mppi
from motion_planning.reachability_planner import ReachabilityPlanner
```

位置2：规划器选择参数（第282行附近，在 `## Robot motion planning` 注释后）
- 新增：
```python
## Planning method selection
PLANNING_METHOD = 'mppi'  # Options: 'mppi' or 'reachability'
# 可以通过命令行参数或配置文件设置
```

位置3：规划器初始化（第288-290行）
- 当前：
```python
Ego_opt = mppi()
Ego_opt.set_params(N, K, K2, x_lim, u_lim, x_dim, uR_dim, dt, L)
Ego_opt.set_cost(Car_cost_R.weights, RH_A, RH_D, W, wac = 1, ca_lat_bd=ca_lat_bd, ca_lon_bd=ca_lon_bd)
```
- 修改为：
```python
if PLANNING_METHOD == 'mppi':
    Ego_opt = mppi()
    Ego_opt.set_params(N, K, K2, x_lim, u_lim, x_dim, uR_dim, dt, L)
    Ego_opt.set_cost(Car_cost_R.weights, RH_A, RH_D, W, wac = 1, ca_lat_bd=ca_lat_bd, ca_lon_bd=ca_lon_bd)
elif PLANNING_METHOD == 'reachability':
    K_R = 1000      # Number of robot candidate sequences
    K_int = 50      # Number of internal state samples
    d_safe = 3.5    # Safety distance threshold
    Ego_opt = ReachabilityPlanner()
    Ego_opt.set_params(N, K_R, K_int, x_lim, u_lim, x_dim, uR_dim, dt, L, d_safe)
    Ego_opt.set_cost(Car_cost_R.weights, RH_A, RH_D, W, wac = 1, ca_lat_bd=ca_lat_bd, ca_lon_bd=ca_lon_bd)
    Ego_opt.set_human_models(dynamics_A, dynamics_D, ilq_results_A, ilq_results_D, beta_w)
```

位置4：规划器调用（第320行）
- 当前：`uR = Ego_opt.solve_mppi(x0, ilq_results_A, ilq_results_D, theta_prob, beta_distr, active, beta_w)`
- 修改为：
```python
if PLANNING_METHOD == 'mppi':
    uR = Ego_opt.solve_mppi(x0, ilq_results_A, ilq_results_D, theta_prob, beta_distr, active, beta_w)
elif PLANNING_METHOD == 'reachability':
    uR = Ego_opt.solve(x0, ilq_results_A, ilq_results_D, theta_prob, beta_distr, active, beta_w)
```

位置5：结果保存（第580行附近）
- 修改 `np.savez_compressed` 调用，添加规划方法标识：
```python
np.savez_compressed(f'{result_dir}/test_{trial}', 
                    ego = Ego_traj, 
                    human = Human_traj, 
                    beta = BETA,
                    t_beta = beta, 
                    theta = THETA, 
                    t_theta = theta,
                    PassInter = PassInter,
                    Collision = Collision,
                    planning_method = PLANNING_METHOD)  # 新增
```

#### 3.2 `motion_planning/MPPI.py` 修改点

位置：`mppi` 类中（可选，用于代码复用）
- 方法：`sample_robot_action_sequences(self, K, N)`（如果不存在）
  - 作用：采样机器人候选控制序列
  - 可被 `ReachabilityPlanner` 复用

注意：如果 `mppi` 类已有采样逻辑，可直接复用；否则在 `ReachabilityPlanner` 中实现。

### 四、接口统一设计

#### 4.1 统一接口规范

两个规划器需实现相同的方法签名：

```python
# 共同接口
class PlannerBase:  # 可选：定义抽象基类
    def set_params(self, ...): pass
    def set_cost(self, ...): pass
    def solve(self, state, ilq_results_A, ilq_results_D, theta_prob, beta_distr, active, beta_w): pass
```

- `mppi.solve_mppi()` → 保持原名或添加 `solve()` 别名
- `ReachabilityPlanner.solve()` → 直接使用 `solve()`

#### 4.2 参数一致性

确保两个规划器接收相同的输入：
- `state`：当前状态 `x0`
- `ilq_results_A/D`：人类决策模型结果
- `theta_prob`：人类特征概率
- `beta_distr`：理性系数分布
- `active`：是否使用 Active Inference（Reachability 中可忽略或用于采样策略）
- `beta_w`：是否考虑理性系数

### 五、实现优先级和步骤

#### Phase 1：基础框架
1. 创建 `internal_state_sampler.py` 并实现 `InternalStateSampler` 类
2. 创建 `reachability_builder.py` 并实现 `ReachabilityBuilder` 类（先实现 `rollout_human_trajectories`）
3. 在 `main.py` 中添加 `PLANNING_METHOD` 参数

#### Phase 2：核心规划器
4. 创建 `reachability_planner.py` 并实现 `ReachabilityPlanner` 类
5. 实现 `build_reachable_sets`（凸包方法）
6. 实现 `solve_safe_mpc_sampling`（采样式求解）

#### Phase 3：集成和测试
7. 在 `main.py` 中完成切换逻辑
8. 测试两种规划方法的对比实验
9. 优化参数和性能

#### Phase 4：可选优化
10. 实现 `safe_mpc_solver.py`（如果使用优化器）
11. 添加椭球可达集方法
12. 添加可视化工具（绘制可达集）

### 六、关键实现细节

#### 6.1 人类动作采样（在 `reachability_builder.py` 中）

需要复用的逻辑：
- 从 `MPPI.py` 的 `_compute_human_action()` 中提取人类动作计算逻辑
- 根据 `psi` 选择对应的 iLQ 结果
- 根据 `beta` 添加高斯噪声：`uH_k = uH_optimal + N(0, Sigma/beta)`

#### 6.2 可达集表示

凸包方法：
- 使用 `scipy.spatial.ConvexHull`
- 存储顶点或半空间表示
- 距离计算：点到凸包的距离

椭球方法（可选）：
- 计算样本的均值和协方差
- 使用椭球 `(x-μ)ᵀΣ⁻¹(x-μ) ≤ 1` 表示
- 距离计算：点到椭球的距离

#### 6.3 安全约束检查

在 `solve_safe_mpc_sampling` 中：
- 对每个时间步 k，检查 `dist(x_R[k], X_H_sets[k]) >= d_safe`
- 如果使用凸包：计算点到凸包的距离
- 如果使用椭球：计算点到椭球的距离

#### 6.4 代价函数复用

- 复用 `mppi.running_cost()` 的逻辑
- 或提取为独立函数 `compute_robot_cost(xR, uR, ...)`

### 七、文件修改清单总结

| 文件路径 | 修改类型 | 具体位置 | 修改内容 |
|---------|---------|---------|---------|
| `motion_planning/internal_state_sampler.py` | 新增 | 全部 | 创建新文件，实现内部状态采样 |
| `motion_planning/reachability_builder.py` | 新增 | 全部 | 创建新文件，实现轨迹仿真和可达集构造 |
| `motion_planning/reachability_planner.py` | 新增 | 全部 | 创建新文件，实现主规划器类 |
| `motion_planning/safe_mpc_solver.py` | 新增（可选） | 全部 | 创建新文件，实现优化器版MPC |
| `main.py` | 修改 | 第19行 | 添加新模块导入 |
| `main.py` | 修改 | 第282行附近 | 添加规划方法选择参数 |
| `main.py` | 修改 | 第288-290行 | 添加条件初始化逻辑 |
| `main.py` | 修改 | 第320行 | 添加条件调用逻辑 |
| `main.py` | 修改 | 第580行附近 | 添加规划方法标识到保存数据 |

### 八、测试和验证方案

#### 8.1 单元测试
- `InternalStateSampler.sample()`：验证采样分布正确性
- `ReachabilityBuilder.rollout_human_trajectories()`：验证轨迹仿真正确性
- `ReachabilityBuilder.build_reachable_sets()`：验证可达集构造正确性

#### 8.2 集成测试
- 在相同初始条件下，对比 `mppi` 和 `ReachabilityPlanner` 的输出
- 验证两种方法都能避免碰撞
- 验证可达集约束的有效性

#### 8.3 对比实验
- 设置 `PLANNING_METHOD = 'mppi'` 和 `'reachability'` 分别运行
- 比较成功率、最小距离、计算时间等指标

---

## 总结

该方案通过新增模块和最小化主循环改动，实现两种规划方法的切换。核心是：
1. 新增 3-4 个模块文件实现 Reachability 规划
2. 在 `main.py` 中添加约 4 处条件判断实现切换
3. 保持接口一致性，便于对比实验
4. 保留所有现有功能，确保向后兼容
