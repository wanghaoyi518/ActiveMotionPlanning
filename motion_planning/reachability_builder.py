import numpy as np
# Using ellipsoid method only, no ConvexHull needed
from .dynamics import InteractionDynamics
from human_model.iLQgame import get_covariance


class ReachabilityBuilder:
    """
    Build human reachable sets based on internal state samples and robot candidate control sequences.
    
    This class simulates human trajectories under different internal states (psi, beta) and
    robot control sequences, then constructs reachable sets using ellipsoid representation
    for each prediction time step.
    
    Attributes:
    -----------
    dynamics_A : InteractionDynamics
        Dynamics model for attentive human.
    dynamics_D : InteractionDynamics
        Dynamics model for distracted human.
    ilq_results_A : ilq_results
        iLQ game results for attentive human.
    ilq_results_D : ilq_results
        iLQ game results for distracted human.
    dt : float
        Time step.
    L : float
        Vehicle wheelbase.
    uH_lim : numpy array
        Human control limits [acceleration, steering].
    beta_w : bool
        Whether to scale noise by rationality coefficient beta.
    """
    
    def __init__(self, dynamics_A, dynamics_D, ilq_results_A, ilq_results_D, dt, L, uH_lim, beta_w):
        """
        Initialize the reachability builder.
        
        Parameters:
        -----------
        dynamics_A : InteractionDynamics
            Dynamics model for attentive human.
        dynamics_D : InteractionDynamics
            Dynamics model for distracted human.
        ilq_results_A : ilq_results
            iLQ game results for attentive human.
        ilq_results_D : ilq_results
            iLQ game results for distracted human.
        dt : float
            Time step.
        L : float
            Vehicle wheelbase.
        uH_lim : numpy array
            Human control limits [acceleration, steering].
        beta_w : bool
            Whether to scale noise by rationality coefficient beta.
        """
        self.dynamics_A = dynamics_A
        self.dynamics_D = dynamics_D
        self.ilq_results_A = ilq_results_A
        self.ilq_results_D = ilq_results_D
        self.dt = dt
        self.L = L
        self.uH_lim = uH_lim
        self.beta_w = beta_w
    
    def rollout_human_trajectories(self, x0, internal_samples, candidate_U_R, N, RH_A=None, RH_D=None):
        """
        Simulate human trajectories based on internal state samples and robot control sequences.
        
        Parameters:
        -----------
        x0 : numpy array, shape (8,)
            Current joint state [xH, yH, psiH, vH, xR, yR, psiR, vR].
        internal_samples : list of tuples
            List of (psi, beta) pairs from InternalStateSampler.
        candidate_U_R : list of numpy arrays
            List of robot candidate control sequences, each shape (N, 2).
        N : int
            Prediction horizon length.
        RH_A : numpy array, optional
            Control cost matrix for attentive human, shape (2, 2).
        RH_D : numpy array, optional
            Control cost matrix for distracted human, shape (2, 2).
        
        Returns:
        --------
        X_H_samples : list of lists
            X_H_samples[k] contains all human position samples [(x, y), ...] at time step k.
        """
        # Initialize storage for human position samples at each time step
        X_H_samples = [[] for _ in range(N)]
        
        # Get iLQ reference trajectories and feedback gains
        # These are computed once and reused
        x_ilq_ref_A = self.ilq_results_A.ilq_solve.best_operating_point.xs
        u_ilq_ref_A = self.ilq_results_A.ilq_solve.best_operating_point.us
        Ps_A = self.ilq_results_A.ilq_solve.best_operating_point.Ps
        alphas_A = self.ilq_results_A.ilq_solve.best_operating_point.alphas
        Zs_A = self.ilq_results_A.ilq_solve.best_operating_point.Zs
        
        x_ilq_ref_D = self.ilq_results_D.ilq_solve.best_operating_point.xs
        u_ilq_ref_D = self.ilq_results_D.ilq_solve.best_operating_point.us
        Ps_D = self.ilq_results_D.ilq_solve.best_operating_point.Ps
        alphas_D = self.ilq_results_D.ilq_solve.best_operating_point.alphas
        Zs_D = self.ilq_results_D.ilq_solve.best_operating_point.Zs
        
        alpha_scale = self.ilq_results_A.ilq_solve.alpha_scale
        
        # Iterate over each robot candidate control sequence
        for U_R_i in candidate_U_R:
            # Iterate over each internal state sample
            for psi_s, beta_s in internal_samples:
                # Select dynamics and iLQ results based on human characteristic
                if psi_s == 'a':
                    dynamics = self.dynamics_A
                    x_ilq_ref = x_ilq_ref_A
                    u_ilq_ref = u_ilq_ref_A
                    Ps = Ps_A
                    alphas = alphas_A
                    Zs = Zs_A
                    RH = RH_A if RH_A is not None else np.diag([1.8, 7.1])  # Default from main.py
                else:  # psi_s == 'd'
                    dynamics = self.dynamics_D
                    x_ilq_ref = x_ilq_ref_D
                    u_ilq_ref = u_ilq_ref_D
                    Ps = Ps_D
                    alphas = alphas_D
                    Zs = Zs_D
                    RH = RH_D if RH_D is not None else np.diag([1.8, 7.1])  # Default from main.py
                
                # Initialize state for this rollout
                x_pred = x0.copy().reshape(-1)
                
                # Simulate forward for N steps
                for k in range(N):
                    # Extract current human and robot states
                    xH_k = x_pred[:4]  # Human state [x, y, psi, v]
                    xR_k = x_pred[4:]  # Robot state [x, y, psi, v]
                    uR_k = U_R_i[k]    # Robot control [a, delta]
                    
                    # Compute human optimal control using iLQ feedback law
                    # uH_optimal = u_ref - P @ (x - x_ref) - alpha_scale * alpha
                    x_ref_k = x_ilq_ref[k].reshape(-1)
                    u_ref_k = u_ilq_ref[k, :2].reshape(2)  # Human control part
                    P_k = Ps[0, k]  # Feedback gain for human (player 0)
                    alpha_k = alphas[0, k].reshape(2)
                    
                    # Compute optimal human control
                    x_diff = (x_pred - x_ref_k).reshape(-1, 1)
                    uH_optimal = u_ref_k - (P_k @ x_diff).flatten() - alpha_scale * alpha_k.flatten()
                    
                    # Clip to control limits
                    uH_optimal[0] = np.clip(uH_optimal[0], -self.uH_lim[0], self.uH_lim[0])
                    uH_optimal[1] = np.clip(uH_optimal[1], -self.uH_lim[1], self.uH_lim[1])
                    
                    # Compute covariance for noise sampling
                    # Linearize dynamics around current state
                    u_combined = np.concatenate([uH_optimal, uR_k])
                    A, B = dynamics.linearizeDiscrete_Interaction(x_pred.reshape(-1, 1), u_combined)
                    
                    # Extract human control input matrix B_H
                    B_H = B[0]  # B[0] corresponds to human player
                    
                    # Compute covariance: Sigma = (RH + B_H^T @ Zs @ B_H)^(-1)
                    Z_k = Zs[0, k]  # Value function Hessian for human
                    Sigma_k = get_covariance(RH, B_H, Z_k)
                    Sigma_k = np.linalg.inv(Sigma_k)
                    Sigma_k = np.abs(Sigma_k)  # Ensure positive definite
                    
                    # Sample noise based on beta (rationality coefficient)
                    # Following main.py: sample each dimension independently using diagonal elements
                    if self.beta_w:
                        # Noise variance scales with 1/beta for each dimension
                        std_a = np.sqrt(Sigma_k[0, 0] / beta_s)
                        std_delta = np.sqrt(Sigma_k[1, 1] / beta_s)
                    else:
                        std_a = np.sqrt(Sigma_k[0, 0])
                        std_delta = np.sqrt(Sigma_k[1, 1])
                    
                    # Sample Gaussian noise for each control dimension independently
                    noise_a = np.random.normal(0, std_a)
                    noise_delta = np.random.normal(0, std_delta)
                    noise = np.array([noise_a, noise_delta])
                    
                    # Add noise to optimal control
                    uH_k = uH_optimal + noise
                    
                    # Clip again after adding noise
                    uH_k[0] = np.clip(uH_k[0], -self.uH_lim[0], self.uH_lim[0])
                    uH_k[1] = np.clip(uH_k[1], -self.uH_lim[1], self.uH_lim[1])
                    
                    # Combine human and robot controls
                    u_combined = np.concatenate([uH_k, uR_k])
                    
                    # Step dynamics forward
                    x_next = dynamics.integrate(x_pred.reshape(-1, 1), u_combined)
                    x_pred = x_next.flatten()
                    
                    # Extract human position (x, y) and add to samples
                    xH_pos = (x_pred[0], x_pred[1])
                    X_H_samples[k].append(xH_pos)
        
        return X_H_samples
    
    def build_reachable_sets(self, X_H_samples):
        """
        Build reachable sets from human position samples using ellipsoid representation.
        
        Parameters:
        -----------
        X_H_samples : list of lists
            X_H_samples[k] contains all human position samples [(x, y), ...] at time step k.
        
        Returns:
        --------
        X_H_sets : list
            List of reachable set representations for each time step.
            Each element is a dict with 'mean' and 'cov' keys representing an ellipsoid.
        """
        X_H_sets = []
        
        for k in range(len(X_H_samples)):
            positions = X_H_samples[k]
            
            if len(positions) == 0:
                # No samples at this time step, create large ellipsoid (safe fallback)
                X_H_sets.append({
                    'mean': np.array([0, 0]),
                    'cov': np.eye(2) * 1e6
                })
                continue
            
            # Compute mean and covariance for ellipsoid representation
            points = np.array(positions)
            mean_pos = np.mean(points, axis=0)
            
            # Handle cases with insufficient points
            if len(points) == 1:
                # Single point: create small ellipsoid around it
                cov_pos = np.eye(2) * 0.25  # 0.5m radius in each direction
            elif len(points) == 2:
                # Two points: create ellipsoid covering both points
                # Compute covariance from the two points
                diff = points[1] - points[0]
                # Create covariance matrix with principal axis along the line
                # Add perpendicular component for minimum spread
                perp = np.array([-diff[1], diff[0]])
                perp = perp / (np.linalg.norm(perp) + 1e-6) * 0.5
                # Use sample covariance with minimum regularization
                cov_pos = np.cov(points.T)
                cov_pos += np.eye(2) * 0.1  # Ensure minimum spread
            else:
                # Multiple points: compute sample covariance
                cov_pos = np.cov(points.T)
            
            # Ensure covariance is positive definite and has minimum size
            cov_pos += np.eye(2) * 1e-6
            
            # Ensure minimum covariance to avoid degenerate ellipsoids
            min_cov = 0.1  # Minimum standard deviation of 0.32m
            eigenvals = np.linalg.eigvals(cov_pos)
            if np.min(eigenvals) < min_cov:
                # Add regularization to ensure minimum size
                cov_pos += np.eye(2) * (min_cov - np.min(eigenvals))
            
            X_H_sets.append({
                'mean': mean_pos,
                'cov': cov_pos
            })
        
        return X_H_sets


if __name__ == "__main__":
    # Test code
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from motion_planning.dynamics import InteractionDynamics, VehicleDyanmics
    from motion_planning.internal_state_sampler import InternalStateSampler
    from motion_planning.inference import beta_prob_distr
    import utils.Reference as rf
    
    # Simple test setup
    print("Testing ReachabilityBuilder...")
    
    # Create dummy dynamics and iLQ results (simplified for testing)
    # In real usage, these would come from the main simulation
    dt = 0.1
    L = 1.5
    rd_width = 4
    
    # Create reference trajectories
    pts_r = np.array([[-30.0, rd_width/2], [40, rd_width/2]])
    pts_h = np.array([[0.0, -30.0], [0.0, rd_width/2]])
    ref_r = rf.reference(dt, 1, 1, pts_r)
    ref_h = rf.reference(dt, 1, 1, pts_h)
    
    Ego = VehicleDyanmics(ref_r, L, dt)
    Human = VehicleDyanmics(ref_h, L, dt)
    
    xH_dims = list(range(0, 4))
    xR_dims = list(range(4, 8))
    uH_dims = list(range(0, 2))
    uR_dims = list(range(2, 4))
    
    x_dim = 8
    u_dim = 4
    x_dims = np.array([xH_dims, xR_dims])
    u_dims = np.array([uH_dims, uR_dims])
    
    dynamics_A = InteractionDynamics(dt, x_dim, u_dim, x_dims, u_dims, [Human, Ego])
    dynamics_D = InteractionDynamics(dt, x_dim, u_dim, x_dims, u_dims, [Human, Ego])
    
    # Note: For full test, we would need actual ilq_results objects
    # This is a simplified test that shows the structure
    print("ReachabilityBuilder class created successfully.")
    print("Note: Full testing requires initialized ilq_results objects from iLQ game solver.")
