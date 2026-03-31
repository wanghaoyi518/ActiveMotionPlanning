import numpy as np
from scipy.stats import truncnorm
from .inference import beta_prob_distr


class InternalStateSampler:
    """
    Sample internal states (psi, beta) from the current belief distribution.
    
    This class samples human characteristics (psi: 'a' for attentive or 'd' for distracted)
    and rationality coefficients (beta: continuous value in [beta_lim[0], beta_lim[1]])
    according to the current belief P(psi, beta | I_t).
    """
    
    def __init__(self, beta_distr, theta_prob):
        """
        Initialize the internal state sampler.
        
        Parameters:
        -----------
        beta_distr : beta_prob_distr
            Object containing beta distributions for both attentive ('a') and distracted ('d') humans.
            Should have attributes:
            - beta_distr.a: prob_distr object with mu, covar, trunc_mu for attentive human
            - beta_distr.d: prob_distr object with mu, covar, trunc_mu for distracted human
            - beta_distr.beta_lim: [beta_min, beta_max] limits for beta values
        theta_prob : list
            [P(attentive), P(distracted)] probabilities for human characteristics.
            Should sum to 1.0.
        """
        self.beta_distr = beta_distr
        self.theta_prob = theta_prob
        
        # Validate theta_prob
        if abs(sum(theta_prob) - 1.0) > 1e-6:
            raise ValueError(f"theta_prob must sum to 1.0, got {sum(theta_prob)}")
        
        if len(theta_prob) != 2:
            raise ValueError(f"theta_prob must have length 2, got {len(theta_prob)}")
    
    def sample(self, K_int):
        """
        Sample K_int internal state pairs (psi, beta).
        
        Parameters:
        -----------
        K_int : int
            Number of internal state samples to generate.
        
        Returns:
        --------
        samples : list of tuples
            List of (psi, beta) pairs, where:
            - psi: str, either 'a' (attentive) or 'd' (distracted)
            - beta: float, rationality coefficient in [beta_lim[0], beta_lim[1]]
        """
        samples = []
        
        # Extract beta limits
        beta_min = self.beta_distr.beta_lim[0]
        beta_max = self.beta_distr.beta_lim[1]
        
        # Sample psi (human characteristic) according to theta_prob
        # theta_prob[0] = P(attentive), theta_prob[1] = P(distracted)
        psi_samples = np.random.choice(['a', 'd'], size=K_int, p=self.theta_prob)
        
        for psi_s in psi_samples:
            # Select the corresponding beta distribution
            if psi_s == 'a':
                beta_dist = self.beta_distr.a
            else:  # psi_s == 'd'
                beta_dist = self.beta_distr.d
            
            # Sample beta from truncated normal distribution
            # Use trunc_mu (truncated mean) and covar (variance) for the distribution
            mu = beta_dist.trunc_mu  # Use truncated mean for sampling
            sigma = np.sqrt(beta_dist.covar)  # Standard deviation
            
            # Create truncated normal distribution
            # truncnorm parameters: (a, b) are standardized bounds
            # a = (beta_min - mu) / sigma, b = (beta_max - mu) / sigma
            if sigma > 1e-6:  # Avoid division by zero
                a_std = (beta_min - mu) / sigma
                b_std = (beta_max - mu) / sigma
                
                # Sample from truncated normal
                beta_s = truncnorm.rvs(a_std, b_std, loc=mu, scale=sigma, size=1)[0]
                
                # Clip to ensure within bounds (numerical safety)
                beta_s = np.clip(beta_s, beta_min, beta_max)
            else:
                # If variance is very small, just use the mean
                beta_s = np.clip(mu, beta_min, beta_max)
            
            samples.append((psi_s, float(beta_s)))
        
        return samples
    
    def update_belief(self, beta_distr, theta_prob):
        """
        Update the belief distribution used for sampling.
        
        Parameters:
        -----------
        beta_distr : beta_prob_distr
            Updated beta distribution object.
        theta_prob : list
            Updated theta probabilities [P(attentive), P(distracted)].
        """
        self.beta_distr = beta_distr
        self.theta_prob = theta_prob
        
        # Validate
        if abs(sum(theta_prob) - 1.0) > 1e-6:
            raise ValueError(f"theta_prob must sum to 1.0, got {sum(theta_prob)}")


if __name__ == "__main__":
    # Test code
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from motion_planning.inference import beta_prob_distr
    
    # Initialize test belief
    mu_beta_init = 0.6
    sigma_beta_init = 0.14
    beta_lim = [0.2, 1.0]
    beta_distr_test = beta_prob_distr(mu_beta_init, sigma_beta_init, beta_lim)
    theta_prob_test = [0.5, 0.5]
    
    # Create sampler
    sampler = InternalStateSampler(beta_distr_test, theta_prob_test)
    
    # Sample
    K_int = 100
    samples = sampler.sample(K_int)
    
    # Print statistics
    print(f"Sampled {len(samples)} internal states:")
    print(f"Attentive samples: {sum(1 for psi, _ in samples if psi == 'a')}")
    print(f"Distracted samples: {sum(1 for psi, _ in samples if psi == 'd')}")
    
    beta_values = [beta for _, beta in samples]
    print(f"Beta statistics:")
    print(f"  Mean: {np.mean(beta_values):.4f}")
    print(f"  Std: {np.std(beta_values):.4f}")
    print(f"  Min: {np.min(beta_values):.4f}")
    print(f"  Max: {np.max(beta_values):.4f}")
    print(f"  Expected range: [{beta_lim[0]}, {beta_lim[1]}]")
    
    # Test with different theta_prob
    print("\nTesting with theta_prob = [0.8, 0.2]:")
    sampler.update_belief(beta_distr_test, [0.8, 0.2])
    samples2 = sampler.sample(100)
    print(f"Attentive samples: {sum(1 for psi, _ in samples2 if psi == 'a')}")
    print(f"Distracted samples: {sum(1 for psi, _ in samples2 if psi == 'd')}")

