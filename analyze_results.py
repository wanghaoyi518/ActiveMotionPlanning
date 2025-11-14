import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def analyze_and_visualize_results(npz_path, result_dir, rd_width=4, rd_length=40):
    """
    Analyze simulation results from npz file and generate comprehensive visualization.
    
    Parameters:
    -----------
    npz_path : str
        Path to the npz file containing simulation results
    result_dir : str
        Directory where analysis results should be saved
    rd_width : float
        Road width (default: 4)
    rd_length : float
        Road length (default: 40)
    """
    # Load results
    data = np.load(npz_path, allow_pickle=True)
    
    ego_traj = data['ego']
    human_traj = data['human']
    beta_est = data['beta']
    theta_est = data['theta']
    true_beta = float(np.array(data['t_beta']).item())
    true_theta = data['t_theta']
    pass_inter = bool(data['PassInter'])
    collision = bool(data['Collision'])
    
    print("\n" + "=" * 60)
    print("Simulation Results Analysis")
    print("=" * 60)
    print(f"True Human Characteristic (theta): {true_theta}")
    print(f"True Human Rationality (beta): {true_beta:.4f}")
    print(f"Successfully Passed Intersection: {pass_inter}")
    print(f"Collision Occurred: {collision}")
    print(f"Total Simulation Steps: {len(ego_traj)}")
    print("=" * 60)
    
    # Convert to numpy arrays
    ego_traj = np.array(ego_traj)
    human_traj = np.array(human_traj)
    beta_est = np.array(beta_est)
    theta_est = np.array(theta_est)
    
    # Fix dimension mismatch: beta_est includes initial value, theta_est doesn't
    if len(beta_est) == len(theta_est) + 1:
        beta_est_plot = beta_est[1:]  # Skip initial value for plotting
        beta_time_steps = np.arange(len(beta_est_plot))
    else:
        beta_est_plot = beta_est
        beta_time_steps = np.arange(len(beta_est_plot))
    
    theta_time_steps = np.arange(len(theta_est))
    traj_time_steps = np.arange(len(ego_traj))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Trajectory plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(human_traj[:, 0], human_traj[:, 1], '-', color='grey', linewidth=2, label='Human Vehicle', alpha=0.7)
    ax1.plot(ego_traj[:, 0], ego_traj[:, 1], '--', color='yellow', linewidth=2, label='Ego Vehicle', alpha=0.7)
    ax1.scatter(human_traj[0, 0], human_traj[0, 1], color='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(human_traj[-1, 0], human_traj[-1, 1], color='red', s=100, marker='s', label='End', zorder=5)
    ax1.scatter(ego_traj[0, 0], ego_traj[0, 1], color='green', s=100, marker='o', zorder=5)
    ax1.scatter(ego_traj[-1, 0], ego_traj[-1, 1], color='red', s=100, marker='s', zorder=5)
    
    # Draw intersection (4-way intersection)
    # Horizontal roads (east-west)
    ax1.hlines(y=rd_width, xmin=-15, xmax=-rd_width/2, color='black', linestyle='solid', linewidth=2)
    ax1.hlines(y=rd_width, xmin=rd_width/2, xmax=rd_length, color='black', linestyle='solid', linewidth=2)
    ax1.hlines(y=0.0, xmin=-15, xmax=-rd_width/2, color='black', linestyle='solid', linewidth=2)
    ax1.hlines(y=0.0, xmin=rd_width/2, xmax=rd_length, color='black', linestyle='solid', linewidth=2)
    
    # Vertical roads (north-south) - lower part
    ax1.vlines(x=-rd_width/2, ymin=-15.0, ymax=0.0, color='black', linestyle='solid', linewidth=2)
    ax1.vlines(x=rd_width/2, ymin=-15.0, ymax=0.0, color='black', linestyle='solid', linewidth=2)
    
    # Vertical roads (north-south) - upper part
    ax1.vlines(x=-rd_width/2, ymin=rd_width, ymax=rd_length, color='black', linestyle='solid', linewidth=2)
    ax1.vlines(x=rd_width/2, ymin=rd_width, ymax=rd_length, color='black', linestyle='solid', linewidth=2)
    
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('Vehicle Trajectories', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 2. Beta estimation over time
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(beta_time_steps, beta_est_plot, 'b-', linewidth=2, label='Estimated Beta', alpha=0.7)
    ax2.axhline(y=true_beta, color='r', linestyle='--', linewidth=2, label=f'True Beta ({true_beta:.4f})')
    ax2.fill_between(beta_time_steps, beta_est_plot - 0.1, beta_est_plot + 0.1, alpha=0.2, color='blue')
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Beta (Rationality)', fontsize=12)
    ax2.set_title('Beta Estimation', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.2, 1.0])
    
    # 3. Beta estimation error
    ax3 = plt.subplot(2, 3, 3)
    beta_error = np.abs(beta_est_plot - true_beta)
    ax3.plot(beta_time_steps, beta_error, 'g-', linewidth=2, label='Estimation Error', alpha=0.7)
    ax3.set_xlabel('Time Step', fontsize=12)
    ax3.set_ylabel('Absolute Error', fontsize=12)
    ax3.set_title('Beta Estimation Error', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. Theta (characteristic) probability over time
    ax4 = plt.subplot(2, 3, 4)
    if true_theta == 'a':
        theta_label = 'Attentive'
        # When true_theta='a', theta_est contains P(Attentive) from THETA array
        ax4.plot(theta_time_steps, theta_est, 'b-', linewidth=2, label=f'P(Attentive)', alpha=0.7)
        ax4.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='True (Attentive)')
    else:
        theta_label = 'Distracted'
        # When true_theta='d', theta_est contains P(Distracted) from THETA array
        # No need to convert: theta_est is already P(Distracted)
        ax4.plot(theta_time_steps, theta_est, 'orange', linewidth=2, label=f'P(Distracted)', alpha=0.7)
        ax4.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='True (Distracted)')
    
    ax4.set_xlabel('Time Step', fontsize=12)
    ax4.set_ylabel('Probability', fontsize=12)
    ax4.set_title(f'Human Characteristic Inference\n(True: {theta_label})', fontsize=14, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.1])
    
    # 5. Distance between vehicles over time
    ax5 = plt.subplot(2, 3, 5)
    distances = np.sqrt((ego_traj[:, 0] - human_traj[:, 0])**2 + (ego_traj[:, 1] - human_traj[:, 1])**2)
    ax5.plot(traj_time_steps, distances, 'purple', linewidth=2, label='Distance', alpha=0.7)
    ax5.axhline(y=3.5, color='r', linestyle='--', linewidth=2, label='Safety Threshold', alpha=0.5)
    ax5.set_xlabel('Time Step', fontsize=12)
    ax5.set_ylabel('Distance (m)', fontsize=12)
    ax5.set_title('Inter-Vehicle Distance', fontsize=14, fontweight='bold')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    
    # 6. Velocity profiles
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(traj_time_steps, ego_traj[:, 3] * 3.6, 'yellow', linewidth=2, label='Ego Velocity', alpha=0.7)
    ax6.plot(traj_time_steps, human_traj[:, 3] * 3.6, 'grey', linewidth=2, label='Human Velocity', alpha=0.7)
    ax6.set_xlabel('Time Step', fontsize=12)
    ax6.set_ylabel('Velocity (km/h)', fontsize=12)
    ax6.set_title('Vehicle Velocities', fontsize=14, fontweight='bold')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    analysis_plot_path = f'{result_dir}/analysis_results.png'
    plt.savefig(analysis_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plot saved to: {analysis_plot_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Key Statistics")
    print("=" * 60)
    print(f"Final Beta Estimation: {beta_est_plot[-1]:.4f}")
    print(f"Beta Estimation Error (Final): {abs(beta_est_plot[-1] - true_beta):.4f}")
    print(f"Mean Beta Estimation Error: {np.mean(beta_error):.4f}")
    print(f"Std Beta Estimation Error: {np.std(beta_error):.4f}")
    print(f"Final Theta Probability: {theta_est[-1]:.4f}")
    print(f"Min Distance Between Vehicles: {np.min(distances):.2f} m")
    print(f"Mean Distance Between Vehicles: {np.mean(distances):.2f} m")
    print(f"Final Ego Velocity: {ego_traj[-1, 3] * 3.6:.2f} km/h")
    print(f"Final Human Velocity: {human_traj[-1, 3] * 3.6:.2f} km/h")
    print("=" * 60)
    
    # Save statistics to txt file
    stats_file_path = f'{result_dir}/key_statistics.txt'
    with open(stats_file_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Key Statistics\n")
        f.write("=" * 60 + "\n")
        f.write(f"Final Beta Estimation: {beta_est_plot[-1]:.4f}\n")
        f.write(f"Beta Estimation Error (Final): {abs(beta_est_plot[-1] - true_beta):.4f}\n")
        f.write(f"Mean Beta Estimation Error: {np.mean(beta_error):.4f}\n")
        f.write(f"Std Beta Estimation Error: {np.std(beta_error):.4f}\n")
        f.write(f"Final Theta Probability: {theta_est[-1]:.4f}\n")
        f.write(f"Min Distance Between Vehicles: {np.min(distances):.2f} m\n")
        f.write(f"Mean Distance Between Vehicles: {np.mean(distances):.2f} m\n")
        f.write(f"Final Ego Velocity: {ego_traj[-1, 3] * 3.6:.2f} km/h\n")
        f.write(f"Final Human Velocity: {human_traj[-1, 3] * 3.6:.2f} km/h\n")
        f.write("=" * 60 + "\n")
    print(f"\nKey statistics saved to: {stats_file_path}")


if __name__ == "__main__":
    # Allow running as standalone script
    import sys
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
        result_dir = sys.argv[2] if len(sys.argv) > 2 else './result'
        rd_width = float(sys.argv[3]) if len(sys.argv) > 3 else 4
        rd_length = float(sys.argv[4]) if len(sys.argv) > 4 else 40
        analyze_and_visualize_results(npz_path, result_dir, rd_width, rd_length)
    else:
        # Default: analyze test_0.npz
        analyze_and_visualize_results('./result/test_0.npz', './result', 4, 40)
