import numpy as np
# Using ellipsoid method only, no ConvexHull needed
from scipy.spatial.distance import cdist
from .internal_state_sampler import InternalStateSampler
from .reachability_builder import ReachabilityBuilder
from .dynamics import InteractionDynamics

# CUDA support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


# Removed distance_point_to_convex_hull function - using ellipsoid method only

def distance_point_to_ellipsoid(point, ellipsoid):
    """
    Compute the distance from a point to an ellipsoid.
    
    Parameters:
    -----------
    point : numpy array, shape (2,)
        Point coordinates [x, y].
    ellipsoid : dict
        Dictionary with 'mean' and 'cov' keys.
    
    Returns:
    --------
    distance : float
        Distance from point to ellipsoid boundary.
    """
    mean = ellipsoid['mean']
    cov = ellipsoid['cov']
    
    # Center point
    centered = point - mean
    
    # Compute Mahalanobis distance
    try:
        cov_inv = np.linalg.inv(cov)
        mahal_dist_sq = centered @ cov_inv @ centered
        mahal_dist = np.sqrt(mahal_dist_sq)
    except:
        # If covariance is singular, use Euclidean distance
        mahal_dist = np.linalg.norm(centered)
    
    # Distance to ellipsoid boundary
    # If mahal_dist < 1, point is inside ellipsoid
    # Approximate distance: (1 - mahal_dist) * sqrt of largest eigenvalue
    if mahal_dist < 1:
        # Point is inside, return negative distance
        eigenvals = np.linalg.eigvals(cov)
        max_eigenval = np.max(eigenvals)
        return -(1 - mahal_dist) * np.sqrt(max_eigenval)
    else:
        # Point is outside, approximate distance
        eigenvals = np.linalg.eigvals(cov)
        max_eigenval = np.max(eigenvals)
        return (mahal_dist - 1) * np.sqrt(max_eigenval)


class ReachabilityPlanner:
    """
    Reachability-based motion planner for safe human-robot interaction.
    
    This planner uses sampling-based reachability analysis to ensure safety
    by maintaining minimum distance from human reachable sets.
    """
    
    def __init__(self, use_cuda=True):
        """
        Initialize the reachability planner.
        
        Parameters:
        -----------
        use_cuda : bool
            Whether to use CUDA acceleration if available.
        """
        self.N = None
        self.K_R = None
        self.K_int = None
        self.dt = None
        self.L = None
        self.d_safe = None
        self.xlim = None
        self.ulim = None
        self.Nx = None
        self.Nu = None
        
        # CUDA support
        self.use_cuda = use_cuda and TORCH_AVAILABLE and torch.cuda.is_available()
        if self.use_cuda:
            self.device = "cuda"
            self.dtype = torch.float64
            print("ReachabilityPlanner: Using CUDA acceleration")
        else:
            self.device = "cpu"
            self.dtype = None
            if use_cuda and not TORCH_AVAILABLE:
                print("ReachabilityPlanner: PyTorch not available, using CPU")
            elif use_cuda and not torch.cuda.is_available():
                print("ReachabilityPlanner: CUDA not available, using CPU")
            else:
                print("ReachabilityPlanner: Using CPU (CUDA disabled)")
        
        # Cost function weights
        self.Qref = None
        self.Qrd = None
        self.Qpsi = None
        self.Qvel = None
        self.Racc = None
        self.Rdel = None
        
        # Human models (set via set_human_models)
        self.dynamics_A = None
        self.dynamics_D = None
        self.ilq_results_A = None
        self.ilq_results_D = None
        self.beta_w = None
        self.RH_A = None
        self.RH_D = None
        
        # Internal components
        self.sampler = None
        self.builder = None
    
    def set_params(self, N, K_R, K_int, xlim, ulim, Nx, Nu, dt, L, d_safe):
        """
        Set planning parameters.
        
        Parameters:
        -----------
        N : int
            Prediction horizon length.
        K_R : int
            Number of robot candidate control sequences to sample.
        K_int : int
            Number of internal state samples.
        xlim : array-like
            State limits.
        ulim : array-like
            Control limits.
        Nx : int
            Total state dimension.
        Nu : int
            Control dimension per agent.
        dt : float
            Time step size.
        L : float
            Vehicle wheelbase length.
        d_safe : float
            Safety distance threshold.
        """
        self.N = N
        self.K_R = K_R
        self.K_int = K_int
        self.dt = dt
        self.L = L
        self.d_safe = d_safe
        self.xlim = np.array(xlim)
        self.ulim = np.array(ulim)
        self.Nx = Nx
        self.Nu = Nu
        
        # Extract robot control limits
        self.uR_min = np.array([-ulim[2], -ulim[3]])
        self.uR_max = np.array([ulim[2], ulim[3]])
        
        # Convert to torch tensors if using CUDA
        if self.use_cuda:
            self.xlim_torch = torch.tensor(xlim, device=self.device, dtype=self.dtype)
            self.uR_min_torch = torch.tensor(self.uR_min, device=self.device, dtype=self.dtype)
            self.uR_max_torch = torch.tensor(self.uR_max, device=self.device, dtype=self.dtype)
    
    def set_cost(self, weight, RH_A, RH_D, W, wac=1, ca_lat_bd=2.5, ca_lon_bd=4):
        """
        Set cost function weights.
        
        Parameters:
        -----------
        weight : list
            Cost weights [Qref, Qrd, Qpsi, Qvel, Racc, Rdel].
        RH_A : numpy array
            Control cost matrix for attentive human.
        RH_D : numpy array
            Control cost matrix for distracted human.
        W : float
            Collision avoidance weight.
        wac : float
            Additional collision avoidance parameter.
        ca_lat_bd : float
            Lateral collision avoidance boundary.
        ca_lon_bd : float
            Longitudinal collision avoidance boundary.
        """
        self.Qref = weight[0]
        self.Qrd = weight[2]
        self.Qpsi = weight[3]
        self.Qvel = weight[4]
        self.Racc = weight[5]
        self.Rdel = weight[6]
        
        self.RH_A = RH_A
        self.RH_D = RH_D
        self.W = W
        self.wac = wac
        self.ca_lat_bd = ca_lat_bd
        self.ca_lon_bd = ca_lon_bd
        
        # Convert cost weights to torch if using CUDA
        if self.use_cuda:
            self.Qref_torch = torch.tensor(self.Qref, device=self.device, dtype=self.dtype)
            self.Qpsi_torch = torch.tensor(self.Qpsi, device=self.device, dtype=self.dtype)
            self.Qvel_torch = torch.tensor(self.Qvel, device=self.device, dtype=self.dtype)
            self.Racc_torch = torch.tensor(self.Racc, device=self.device, dtype=self.dtype)
            self.Rdel_torch = torch.tensor(self.Rdel, device=self.device, dtype=self.dtype)
            self.d_safe_torch = torch.tensor(self.d_safe, device=self.device, dtype=self.dtype)
    
    def set_human_models(self, dynamics_A, dynamics_D, ilq_results_A, ilq_results_D, beta_w):
        """
        Set human decision-making models.
        
        Parameters:
        -----------
        dynamics_A : InteractionDynamics
            Dynamics for attentive human.
        dynamics_D : InteractionDynamics
            Dynamics for distracted human.
        ilq_results_A : ilq_results
            iLQ results for attentive human.
        ilq_results_D : ilq_results
            iLQ results for distracted human.
        beta_w : bool
            Whether to consider rationality coefficient.
        """
        self.dynamics_A = dynamics_A
        self.dynamics_D = dynamics_D
        self.ilq_results_A = ilq_results_A
        self.ilq_results_D = ilq_results_D
        self.beta_w = beta_w
        
        # Initialize reachability builder
        uH_lim = [self.ulim[0], self.ulim[1]]
        self.builder = ReachabilityBuilder(
            dynamics_A, dynamics_D, ilq_results_A, ilq_results_D,
            self.dt, self.L, uH_lim, beta_w
        )
    
    def sample_robot_control_sequences(self, K_R, N):
        """
        Sample robot candidate control sequences.
        
        Parameters:
        -----------
        K_R : int
            Number of sequences to sample.
        N : int
            Horizon length.
        
        Returns:
        --------
        candidate_U_R : list or torch.Tensor
            If CUDA: torch.Tensor shape (K_R, N, 2)
            If CPU: list of control sequences, each shape (N, 2).
        """
        if self.use_cuda:
            # Batch sampling on GPU
            candidate_U_R = torch.distributions.Uniform(
                low=self.uR_min_torch,
                high=self.uR_max_torch
            ).sample((K_R, N))
            return candidate_U_R
        else:
            # CPU version (original)
            candidate_U_R = []
            for _ in range(K_R):
                u_seq = np.random.uniform(
                    low=self.uR_min,
                    high=self.uR_max,
                    size=(N, 2)
                )
                candidate_U_R.append(u_seq)
            return candidate_U_R
    
    def compute_robot_cost(self, xR_traj, uR_seq):
        """
        Compute cost for a robot trajectory.
        
        Parameters:
        -----------
        xR_traj : numpy array or torch.Tensor, shape (N+1, 4) or (K_R, N+1, 4)
            Robot state trajectory [x, y, psi, v] for N+1 steps.
            If batch mode: shape (K_R, N+1, 4)
        uR_seq : numpy array or torch.Tensor, shape (N, 2) or (K_R, N, 2)
            Robot control sequence [a, delta] for N steps.
            If batch mode: shape (K_R, N, 2)
        
        Returns:
        --------
        cost : float or torch.Tensor
            Total trajectory cost. If batch mode: shape (K_R,)
        """
        if self.use_cuda and torch.is_tensor(xR_traj):
            # Batch CUDA version
            K_R = xR_traj.shape[0]
            costs = torch.zeros((K_R,), device=self.device, dtype=self.dtype)
            
            for k in range(self.N):
                xR_k = xR_traj[:, k, :]  # (K_R, 4)
                uR_k = uR_seq[:, k, :]   # (K_R, 2)
                
                # Vectorized cost computation
                costs += self.Qref_torch * ((xR_k[:, 1] - self.xlim_torch[1] / 2)**2)
                costs += self.Qpsi_torch * ((xR_k[:, 2] - 0.0)**2)
                costs += self.Qvel_torch * ((xR_k[:, 3] - self.xlim_torch[3])**2)
                costs += self.Racc_torch * (uR_k[:, 0]**2) + self.Rdel_torch * (uR_k[:, 1]**2)
            
            return costs
        else:
            # CPU version (original)
            cost = 0.0
            
            for k in range(self.N):
                xR_k = xR_traj[k]
                uR_k = uR_seq[k]
                
                # Reference tracking cost
                cost += self.Qref * ((xR_k[1] - self.xlim[1] / 2)**2)  # Center line deviation
                cost += self.Qpsi * ((xR_k[2] - 0.0)**2)  # Heading deviation
                cost += self.Qvel * ((xR_k[3] - self.xlim[3])**2)  # Velocity deviation
                
                # Control penalty
                cost += self.Racc * (uR_k[0]**2) + self.Rdel * (uR_k[1]**2)
            
            return cost
    
    def batch_simulate_robot_trajectories(self, U_R_all, xR_0):
        """
        Batch simulate robot trajectories (CUDA optimized).
        
        Parameters:
        -----------
        U_R_all : torch.Tensor, shape (K_R, N, 2)
            Batch of robot control sequences.
        xR_0 : torch.Tensor, shape (4,)
            Initial robot state.
        
        Returns:
        --------
        xR_traj_all : torch.Tensor, shape (K_R, N+1, 4)
            Batch of robot trajectories.
        """
        K_R, N, _ = U_R_all.shape
        
        # Initialize trajectory tensor
        xR_traj = torch.zeros((K_R, N+1, 4), device=self.device, dtype=self.dtype)
        xR_traj[:, 0] = xR_0.unsqueeze(0).repeat(K_R, 1)
        
        # Batch simulation (vectorized)
        for k in range(N):
            uR_k = U_R_all[:, k, :]  # (K_R, 2)
            xR_k = xR_traj[:, k, :]  # (K_R, 4)
            
            # Vectorized bicycle model update
            delta = uR_k[:, 1]  # (K_R,)
            a = uR_k[:, 0]      # (K_R,)
            v = xR_k[:, 3]      # (K_R,)
            psi = xR_k[:, 2]    # (K_R,)
            
            xR_traj[:, k+1, 0] = xR_k[:, 0] + v * torch.cos(psi) * self.dt
            xR_traj[:, k+1, 1] = xR_k[:, 1] + v * torch.sin(psi) * self.dt
            xR_traj[:, k+1, 2] = xR_k[:, 2] + v * torch.tan(delta) * self.dt / self.L
            xR_traj[:, k+1, 3] = xR_k[:, 3] + a * self.dt
        
        return xR_traj
    
    def batch_check_safety_constraints(self, xR_traj_all, X_H_sets):
        """
        Batch check safety constraints (CUDA optimized).
        
        Parameters:
        -----------
        xR_traj_all : torch.Tensor, shape (K_R, N+1, 4)
            Batch of robot trajectories.
        X_H_sets : list
            Human reachable sets (ellipsoids) for each time step.
        
        Returns:
        --------
        is_safe_all : torch.Tensor, shape (K_R,), bool
            Safety flags for each trajectory.
        min_distances : torch.Tensor, shape (K_R,)
            Minimum distances for each trajectory.
        violation_counts : torch.Tensor, shape (K_R,)
            Violation counts for each trajectory.
        """
        K_R, N_plus_1, _ = xR_traj_all.shape
        N = N_plus_1 - 1
        
        # Convert ellipsoids to tensors
        means = []
        covs = []
        for X_H_k in X_H_sets:
            if isinstance(X_H_k, dict):
                means.append(torch.tensor(X_H_k['mean'], device=self.device, dtype=self.dtype))
                covs.append(torch.tensor(X_H_k['cov'], device=self.device, dtype=self.dtype))
            else:
                # Fallback: use large ellipsoid
                means.append(torch.tensor([0.0, 0.0], device=self.device, dtype=self.dtype))
                covs.append(torch.eye(2, device=self.device, dtype=self.dtype) * 1e6)
        
        means = torch.stack(means)  # (N, 2)
        covs = torch.stack(covs)    # (N, 2, 2)
        
        # Initialize distance tracking
        min_distances = torch.full((K_R,), float('inf'), device=self.device, dtype=self.dtype)
        violation_counts = torch.zeros((K_R,), device=self.device, dtype=torch.long)
        
        # Batch compute distances for all time steps
        for k in range(N):
            xR_pos_k = xR_traj_all[:, k, :2]  # (K_R, 2)
            mean_k = means[k]  # (2,)
            cov_k = covs[k]    # (2, 2)
            
            # Batch Mahalanobis distance computation
            centered = xR_pos_k - mean_k.unsqueeze(0)  # (K_R, 2)
            
            try:
                cov_inv = torch.inverse(cov_k)  # (2, 2)
                # Batch matrix multiplication: (K_R, 2) @ (2, 2) @ (K_R, 2)^T
                mahal_dist_sq = torch.sum(centered @ cov_inv * centered, dim=1)  # (K_R,)
                mahal_dist = torch.sqrt(mahal_dist_sq + 1e-10)  # (K_R,)
            except:
                # If singular, use Euclidean distance
                mahal_dist = torch.norm(centered, dim=1)  # (K_R,)
            
            # Compute distance to ellipsoid boundary
            eigenvals = torch.linalg.eigvals(cov_k).real
            max_eigenval = torch.max(eigenvals)
            
            # Batch distance computation
            dist_k = torch.where(
                mahal_dist < 1.0,
                -(1.0 - mahal_dist) * torch.sqrt(max_eigenval),
                (mahal_dist - 1.0) * torch.sqrt(max_eigenval)
            )  # (K_R,)
            
            # Update minimum distances
            min_distances = torch.minimum(min_distances, dist_k)
            
            # Count violations
            violation_counts += (dist_k < self.d_safe_torch).long()
        
        # Determine safety
        is_safe_all = (violation_counts == 0) & (min_distances >= self.d_safe_torch)
        
        return is_safe_all, min_distances, violation_counts
    
    def batch_check_state_constraints(self, xR_traj_all):
        """
        Batch check state constraints (CUDA optimized).
        
        Parameters:
        -----------
        xR_traj_all : torch.Tensor, shape (K_R, N+1, 4)
            Batch of robot trajectories.
        
        Returns:
        --------
        state_violations : torch.Tensor, shape (K_R,), bool
            State violation flags for each trajectory.
        """
        # Check road boundaries and velocity limits
        y_violation = (xR_traj_all[:, :, 1] < 0.0) | (xR_traj_all[:, :, 1] > self.xlim_torch[1])
        v_violation = (xR_traj_all[:, :, 3] < 0.0) | (xR_traj_all[:, :, 3] > self.xlim_torch[7])
        
        # Any violation at any time step marks the trajectory as infeasible
        state_violations = torch.any(y_violation | v_violation, dim=1)
        
        return state_violations
    
    def check_safety_constraints(self, xR_traj, X_H_sets):
        """
        Check if robot trajectory satisfies safety constraints.
        
        Parameters:
        -----------
        xR_traj : numpy array, shape (N+1, 4)
            Robot state trajectory.
        X_H_sets : list
            Human reachable sets for each time step.
        
        Returns:
        --------
        is_safe : bool
            True if trajectory is safe.
        min_distance : float
            Minimum distance to human reachable sets.
        violation_count : int
            Number of time steps violating safety constraint.
        """
        min_distance = np.inf
        violation_count = 0
        
        for k in range(self.N):
            xR_pos = xR_traj[k, :2]  # Robot position [x, y]
            X_H_k = X_H_sets[k]
            
            # Compute distance to ellipsoid (only method used)
            if isinstance(X_H_k, dict):
                dist = distance_point_to_ellipsoid(xR_pos, X_H_k)
            else:
                # Fallback: use large distance
                dist = np.inf
            
            min_distance = min(min_distance, dist)
            
            if dist < self.d_safe:
                violation_count += 1
        
        is_safe = (violation_count == 0) and (min_distance >= self.d_safe)
        return is_safe, min_distance, violation_count
    
    def solve_safe_mpc_sampling(self, x0, X_H_sets, candidate_U_R):
        """
        Solve safe MPC using sampling-based approach (CUDA optimized if available).
        
        Parameters:
        -----------
        x0 : numpy array, shape (8,)
            Current joint state.
        X_H_sets : list
            Human reachable sets for each time step.
        candidate_U_R : list or torch.Tensor
            Candidate robot control sequences.
            If CUDA: torch.Tensor shape (K_R, N, 2)
            If CPU: list of arrays, each shape (N, 2)
        
        Returns:
        --------
        u_r_t : numpy array, shape (2,)
            Optimal control for current time step.
        """
        # Extract robot initial state
        xR_0 = x0[4:8]
        
        if self.use_cuda and torch.is_tensor(candidate_U_R):
            # CUDA optimized batch processing
            return self._solve_safe_mpc_cuda(xR_0, X_H_sets, candidate_U_R)
        else:
            # CPU version (original implementation)
            return self._solve_safe_mpc_cpu(xR_0, X_H_sets, candidate_U_R)
    
    def _solve_safe_mpc_cuda(self, xR_0, X_H_sets, U_R_all):
        """
        CUDA optimized version of safe MPC solver.
        
        Parameters:
        -----------
        xR_0 : numpy array, shape (4,)
            Initial robot state.
        X_H_sets : list
            Human reachable sets for each time step.
        U_R_all : torch.Tensor, shape (K_R, N, 2)
            Batch of robot control sequences.
        
        Returns:
        --------
        u_r_t : numpy array, shape (2,)
            Optimal control for current time step.
        """
        # Convert initial state to tensor
        xR_0_torch = torch.tensor(xR_0, device=self.device, dtype=self.dtype)
        
        # Step 1: Batch simulate robot trajectories
        xR_traj_all = self.batch_simulate_robot_trajectories(U_R_all, xR_0_torch)
        
        # Step 2: Batch check safety constraints
        is_safe_all, min_distances, violation_counts = \
            self.batch_check_safety_constraints(xR_traj_all, X_H_sets)
        
        # Step 3: Batch check state constraints
        state_violations = self.batch_check_state_constraints(xR_traj_all)
        
        # Step 4: Filter feasible sequences
        feasible_mask = is_safe_all & (~state_violations)
        
        if torch.any(feasible_mask):
            # Have feasible solutions: compute costs and select best
            costs_all = self.compute_robot_cost(xR_traj_all, U_R_all)
            
            # Set infeasible sequences to infinite cost
            feasible_costs = torch.where(
                feasible_mask,
                costs_all,
                torch.tensor(float('inf'), device=self.device, dtype=self.dtype)
            )
            
            best_idx = torch.argmin(feasible_costs).item()
            best_sequence = U_R_all[best_idx]
        else:
            # No feasible solution: select sequence with minimum violations
            # Combine safety and state violations
            total_violations = violation_counts + state_violations.long()
            best_idx = torch.argmin(total_violations).item()
            best_sequence = U_R_all[best_idx]
        
        # Return first control action (convert to numpy)
        return best_sequence[0].cpu().numpy()
    
    def _solve_safe_mpc_cpu(self, xR_0, X_H_sets, candidate_U_R):
        """
        CPU version of safe MPC solver (original implementation).
        
        Parameters:
        -----------
        xR_0 : numpy array, shape (4,)
            Initial robot state.
        X_H_sets : list
            Human reachable sets for each time step.
        candidate_U_R : list
            Candidate robot control sequences.
        
        Returns:
        --------
        u_r_t : numpy array, shape (2,)
            Optimal control for current time step.
        """
        feasible_sequences = []
        feasible_costs = []
        feasible_distances = []
        
        # Check each candidate sequence
        for U_R_i in candidate_U_R:
            # Simulate robot trajectory
            xR_traj = np.zeros((self.N + 1, 4))
            xR_traj[0] = xR_0.copy()
            
            # Use robot dynamics to propagate
            for k in range(self.N):
                uR_k = U_R_i[k]
                xR_k = xR_traj[k]
                
                # Simple bicycle model update
                delta = uR_k[1]
                a = uR_k[0]
                
                x_next = xR_k[0] + xR_k[3] * np.cos(xR_k[2]) * self.dt
                y_next = xR_k[1] + xR_k[3] * np.sin(xR_k[2]) * self.dt
                psi_next = xR_k[2] + xR_k[3] * np.tan(delta) * self.dt / self.L
                v_next = xR_k[3] + a * self.dt
                
                xR_traj[k + 1] = [x_next, y_next, psi_next, v_next]
            
            # Check safety constraints
            is_safe, min_dist, violation_count = self.check_safety_constraints(xR_traj, X_H_sets)
            
            # Check state constraints (road boundaries, velocity limits)
            state_violation = False
            for k in range(self.N + 1):
                if (xR_traj[k, 1] < 0.0 or xR_traj[k, 1] > self.xlim[1] or
                    xR_traj[k, 3] < 0.0 or xR_traj[k, 3] > self.xlim[7]):
                    state_violation = True
                    break
            
            if is_safe and not state_violation:
                # Compute cost
                cost = self.compute_robot_cost(xR_traj, U_R_i)
                feasible_sequences.append(U_R_i)
                feasible_costs.append(cost)
                feasible_distances.append(min_dist)
        
        # Select best sequence
        if len(feasible_sequences) > 0:
            # Choose sequence with minimum cost
            best_idx = np.argmin(feasible_costs)
            best_sequence = feasible_sequences[best_idx]
        else:
            # No feasible sequence found, use fallback strategy
            best_sequence = None
            min_violations = np.inf
            
            for U_R_i in candidate_U_R:
                # Simulate robot trajectory
                xR_traj = np.zeros((self.N + 1, 4))
                xR_traj[0] = xR_0.copy()
                
                for k in range(self.N):
                    uR_k = U_R_i[k]
                    xR_k = xR_traj[k]
                    
                    delta = np.clip(uR_k[1], -self.ulim[3], self.ulim[3])
                    a = np.clip(uR_k[0], -self.ulim[2], self.ulim[2])
                    
                    x_next = xR_k[0] + xR_k[3] * np.cos(xR_k[2]) * self.dt
                    y_next = xR_k[1] + xR_k[3] * np.sin(xR_k[2]) * self.dt
                    psi_next = xR_k[2] + xR_k[3] * np.tan(delta) * self.dt / self.L
                    v_next = np.clip(xR_k[3] + a * self.dt, 0, self.xlim[7])
                    
                    xR_traj[k + 1] = [x_next, y_next, psi_next, v_next]
                
                _, _, violation_count = self.check_safety_constraints(xR_traj, X_H_sets)
                
                if violation_count < min_violations:
                    min_violations = violation_count
                    best_sequence = U_R_i
            
            if best_sequence is None:
                best_sequence = np.zeros((self.N, 2))
        
        # Return first control action
        return best_sequence[0]
    
    def solve(self, state, ilq_results_A, ilq_results_D, theta_prob, beta_distr, active, beta_w):
        """
        Solve reachability-based planning problem.
        
        Parameters:
        -----------
        state : numpy array, shape (8,)
            Current joint state [xH, yH, psiH, vH, xR, yR, psiR, vR].
        ilq_results_A : ilq_results
            iLQ results for attentive human (updated).
        ilq_results_D : ilq_results
            iLQ results for distracted human (updated).
        theta_prob : list
            [P(attentive), P(distracted)] probabilities.
        beta_distr : beta_prob_distr
            Beta distribution object.
        active : bool
            Whether using active inference (not used in reachability planning).
        beta_w : bool
            Whether to consider rationality coefficient.
        
        Returns:
        --------
        u_r_t : numpy array, shape (2,)
            Optimal robot control for current time step.
        """
        # Update human models if needed
        if self.ilq_results_A != ilq_results_A or self.ilq_results_D != ilq_results_D:
            self.set_human_models(
                self.dynamics_A, self.dynamics_D,
                ilq_results_A, ilq_results_D, beta_w
            )
        
        # Step 1: Sample internal states
        self.sampler = InternalStateSampler(beta_distr, theta_prob)
        internal_samples = self.sampler.sample(self.K_int)
        
        # Step 2: Sample robot candidate control sequences
        candidate_U_R = self.sample_robot_control_sequences(self.K_R, self.N)
        
        # Step 3: Rollout human trajectories and build reachable sets
        # Convert torch.Tensor to list for rollout_human_trajectories if needed
        if self.use_cuda and torch.is_tensor(candidate_U_R):
            candidate_U_R_list = [U_R_i.cpu().numpy() for U_R_i in candidate_U_R]
        else:
            candidate_U_R_list = candidate_U_R
        
        X_H_samples = self.builder.rollout_human_trajectories(
            state, internal_samples, candidate_U_R_list, self.N,
            RH_A=self.RH_A, RH_D=self.RH_D
        )
        
        # Step 4: Build reachable sets (using ellipsoid method)
        X_H_sets = self.builder.build_reachable_sets(X_H_samples)
        
        # Step 5: Solve safe MPC
        # Use original candidate_U_R (torch.Tensor if CUDA) for MPC solver
        u_r_t = self.solve_safe_mpc_sampling(state, X_H_sets, candidate_U_R)
        
        return u_r_t


if __name__ == "__main__":
    # Test code
    print("ReachabilityPlanner class created successfully.")
    print("Note: Full testing requires integration with main simulation loop.")

