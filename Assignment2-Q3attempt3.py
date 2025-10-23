import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm



def generate_landmarks(K):

    """Generate K evenly spaced landmarks on unit circle"""

    angles = np.linspace(0, 2*np.pi, K, endpoint=False)

    landmarks = np.column_stack([np.cos(angles), np.sin(angles)])

    return landmarks



def generate_true_position():

    """Generate random true position inside unit circle"""

    r = np.sqrt(np.random.rand()) * 0.8  # 0.8 to keep well inside

    theta = np.random.rand() * 2 * np.pi

    x_true = r * np.cos(theta)

    y_true = r * np.sin(theta)

    return np.array([x_true, y_true])



def generate_measurements(true_pos, landmarks, sigma_measurement):

    """Generate range measurements with Gaussian noise"""

    K = len(landmarks)

    measurements = []

   

    for i in range(K):

        # True distance

        d_true = np.linalg.norm(true_pos - landmarks[i])

       

        # Add Gaussian noise (reject if negative)

        measurement = -1

        while measurement < 0:

            noise = np.random.randn() * sigma_measurement

            measurement = d_true + noise

       

        measurements.append(measurement)

   

    return np.array(measurements)



def map_objective(x, y, landmarks, measurements, sigma_x, sigma_y, sigma_measurement):

    """

    Compute MAP objective function J(x,y)

   

    J(x,y) = sum_i [(r_i - d_i(x,y))^2 / (2*sigma_i^2)] + x^2/(2*sigma_x^2) + y^2/(2*sigma_y^2)

    """

    K = len(landmarks)

    objective = 0.0

   

    # Likelihood term

    for i in range(K):

        predicted_dist = np.sqrt((x - landmarks[i,0])**2 + (y - landmarks[i,1])**2)

        residual = measurements[i] - predicted_dist

        objective += (residual**2) / (2 * sigma_measurement**2)

   

    # Prior term

    objective += (x**2) / (2 * sigma_x**2)

    objective += (y**2) / (2 * sigma_y**2)

   

    return objective



def compute_map_estimate(landmarks, measurements, sigma_x, sigma_y, sigma_measurement):

    """Find MAP estimate using grid search (for visualization purposes)"""

    x_range = np.linspace(-2, 2, 100)

    y_range = np.linspace(-2, 2, 100)

   

    min_obj = float('inf')

    x_map = 0

    y_map = 0

   

    for x in x_range:

        for y in y_range:

            obj = map_objective(x, y, landmarks, measurements,

                              sigma_x, sigma_y, sigma_measurement)

            if obj < min_obj:

                min_obj = obj

                x_map = x

                y_map = y

   

    return np.array([x_map, y_map])



def plot_map_contours(K, true_pos, landmarks, measurements,

                     sigma_x, sigma_y, sigma_measurement,

                     save_name=None):

    """Plot MAP objective function contours"""

   

    # Create grid

    x_range = np.linspace(-2, 2, 200)

    y_range = np.linspace(-2, 2, 200)

    X, Y = np.meshgrid(x_range, y_range)

   

    # Compute objective function on grid

    Z = np.zeros_like(X)

    for i in range(len(x_range)):

        for j in range(len(y_range)):

            Z[j, i] = map_objective(X[j,i], Y[j,i], landmarks, measurements,

                                   sigma_x, sigma_y, sigma_measurement)

   

    # Find MAP estimate

    map_estimate = compute_map_estimate(landmarks, measurements,

                                       sigma_x, sigma_y, sigma_measurement)

   

    # Create figure

    plt.figure(figsize=(10, 9))

   

    # Plot contours at consistent levels for comparison across K values

    # Use levels that work well across different K

    levels = np.linspace(np.min(Z), np.min(Z) + 20, 15)

   

    contour = plt.contour(X, Y, Z, levels=levels, cmap='viridis', linewidths=1.5)

    plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

   

    # Filled contours for better visualization

    plt.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)

   

    # Plot true position with + marker

    plt.plot(true_pos[0], true_pos[1], 'r+', markersize=20,

            markeredgewidth=3, label='True Position')

   

    # Plot landmarks with o marker

    plt.plot(landmarks[:,0], landmarks[:,1], 'go', markersize=12,

            markeredgewidth=2, markerfacecolor='lime', label='Landmarks')

   

    # Plot MAP estimate

    plt.plot(map_estimate[0], map_estimate[1], 'b*', markersize=20,

            markeredgewidth=2, label='MAP Estimate')

   

    # Add unit circle for reference

    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='gray',

                       linestyle='--', linewidth=1.5, label='Unit Circle')

    plt.gca().add_patch(circle)

   

    plt.xlim(-2, 2)

    plt.ylim(-2, 2)

    plt.xlabel('x', fontsize=14)

    plt.ylabel('y', fontsize=14)

    plt.title(f'MAP Objective Function Contours (K = {K})', fontsize=16, fontweight='bold')

    plt.legend(fontsize=11, loc='upper right')

    plt.grid(True, alpha=0.3)

    plt.axis('equal')

    plt.colorbar(contour, label='Objective Value')

   

    # Add text with error

    error = np.linalg.norm(true_pos - map_estimate)

    plt.text(-1.9, 1.8, f'Localization Error: {error:.3f}',

            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

   

    plt.tight_layout()

   

    if save_name:

        plt.savefig(save_name, dpi=300, bbox_inches='tight')

   

    plt.show()

   

    return map_estimate, error



def main():

    """Main function to run the complete analysis"""

   

    # Set random seed for reproducibility

    np.random.seed(42)

   

    # Parameters

    sigma_x = 0.25

    sigma_y = 0.25

    sigma_measurement = 0.3

   

    # Generate true position once

    true_pos = generate_true_position()

   

    print("=" * 70)

    print("Vehicle Localization using MAP Estimation")

    print("=" * 70)

    print(f"\nTrue Vehicle Position: ({true_pos[0]:.4f}, {true_pos[1]:.4f})")

    print(f"Prior Standard Deviations: σ_x = σ_y = {sigma_x}")

    print(f"Measurement Noise Std Dev: σ_measurement = {sigma_measurement}")

    print("\n" + "=" * 70 + "\n")

   

    # Results storage

    results = []

    all_data = []  # Store data for combined plot

   

    # Loop through different values of K

    for K in range(1, 5):

        print(f"\n{'='*70}")

        print(f"Analysis for K = {K} landmarks")

        print(f"{'='*70}")

       

        # Generate landmarks

        landmarks = generate_landmarks(K)

        print(f"\nLandmark positions:")

        for i, lm in enumerate(landmarks):

            print(f"  Landmark {i+1}: ({lm[0]:.4f}, {lm[1]:.4f})")

       

        # Generate measurements

        measurements = generate_measurements(true_pos, landmarks, sigma_measurement)

        print(f"\nRange measurements:")

        for i, r in enumerate(measurements):

            true_dist = np.linalg.norm(true_pos - landmarks[i])

            print(f"  r_{i+1} = {r:.4f} (true distance: {true_dist:.4f})")

       

        # Compute MAP estimate and objective function grid

        map_estimate = compute_map_estimate(landmarks, measurements,

                                           sigma_x, sigma_y, sigma_measurement)

        error = np.linalg.norm(true_pos - map_estimate)

       

        # Create grid for contour plot

        x_range = np.linspace(-2, 2, 200)

        y_range = np.linspace(-2, 2, 200)

        X, Y = np.meshgrid(x_range, y_range)

       

        # Compute objective function on grid

        Z = np.zeros_like(X)

        for i in range(len(x_range)):

            for j in range(len(y_range)):

                Z[j, i] = map_objective(X[j,i], Y[j,i], landmarks, measurements,

                                       sigma_x, sigma_y, sigma_measurement)

       

        print(f"\nMAP Estimate: ({map_estimate[0]:.4f}, {map_estimate[1]:.4f})")

        print(f"Localization Error: {error:.4f}")

       

        results.append({

            'K': K,

            'map_estimate': map_estimate,

            'error': error

        })

       

        all_data.append({

            'K': K,

            'X': X,

            'Y': Y,

            'Z': Z,

            'landmarks': landmarks,

            'measurements': measurements,

            'map_estimate': map_estimate,

            'error': error

        })

   

    # Create combined figure with 2x2 subplots

    print(f"\n{'='*70}")

    print("Creating combined visualization...")

    print(f"{'='*70}\n")

   

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    axes = axes.flatten()

   

    for idx, data in enumerate(all_data):

        ax = axes[idx]

        K = data['K']

        X, Y, Z = data['X'], data['Y'], data['Z']

        landmarks = data['landmarks']

        map_estimate = data['map_estimate']

        error = data['error']

       

        # Plot contours at consistent levels

        levels = np.linspace(np.min(Z), np.min(Z) + 20, 15)

       

        contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', linewidths=1.5)

        ax.clabel(contour, inline=True, fontsize=7, fmt='%.1f')

       

        # Filled contours

        ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)

       

        # Plot true position with + marker

        ax.plot(true_pos[0], true_pos[1], 'r+', markersize=18,

                markeredgewidth=3, label='True Position')

       

        # Plot landmarks with o marker

        ax.plot(landmarks[:,0], landmarks[:,1], 'go', markersize=11,

                markeredgewidth=2, markerfacecolor='lime', label='Landmarks')

       

        # Plot MAP estimate

        ax.plot(map_estimate[0], map_estimate[1], 'b*', markersize=18,

                markeredgewidth=2, label='MAP Estimate')

       

        # Add unit circle

        circle = plt.Circle((0, 0), 1, fill=False, edgecolor='gray',

                           linestyle='--', linewidth=1.5, alpha=0.5)

        ax.add_patch(circle)

       

        ax.set_xlim(-2, 2)

        ax.set_ylim(-2, 2)

        ax.set_xlabel('x', fontsize=12, fontweight='bold')

        ax.set_ylabel('y', fontsize=12, fontweight='bold')

        ax.set_title(f'K = {K} Landmarks (Error: {error:.3f})',

                    fontsize=13, fontweight='bold')

        ax.legend(fontsize=9, loc='upper right')

        ax.grid(True, alpha=0.3)

        ax.set_aspect('equal')

   

    plt.suptitle('MAP Objective Function Contours for Different K Values',

                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    plt.savefig('all_contours_combined.png', dpi=300, bbox_inches='tight')

    plt.show()

   

    # Summary plot

    print(f"\n{'='*70}")

    print("Creating summary plots...")

    print(f"{'='*70}\n")

   

    plt.figure(figsize=(14, 6))

   

    # Plot error vs K

    K_values = [r['K'] for r in results]

    errors = [r['error'] for r in results]

   

    plt.subplot(1, 2, 1)

    plt.plot(K_values, errors, 'bo-', linewidth=2, markersize=12)

    for i, (k, e) in enumerate(zip(K_values, errors)):

        plt.text(k, e + 0.01, f'{e:.3f}', ha='center', va='bottom', fontsize=10)

    plt.xlabel('Number of Landmarks (K)', fontsize=13, fontweight='bold')

    plt.ylabel('Localization Error', fontsize=13, fontweight='bold')

    plt.title('MAP Estimation Error vs K', fontsize=14, fontweight='bold')

    plt.grid(True, alpha=0.3)

    plt.xticks(K_values)

   

    # Plot MAP estimates

    plt.subplot(1, 2, 2)

    for r in results:

        plt.plot(r['map_estimate'][0], r['map_estimate'][1], 'o',

                markersize=14, label=f"K={r['K']}", markeredgewidth=2)

    plt.plot(true_pos[0], true_pos[1], 'r*', markersize=22,

            markeredgewidth=2, label='True Position')

   

    # Add unit circle

    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='gray',

                       linestyle='--', linewidth=2, alpha=0.5)

    plt.gca().add_patch(circle)

   

    plt.xlabel('x', fontsize=13, fontweight='bold')

    plt.ylabel('y', fontsize=13, fontweight='bold')

    plt.title('MAP Estimates for Different K', fontsize=14, fontweight='bold')

    plt.legend(fontsize=11, loc='best')

    plt.grid(True, alpha=0.3)

    plt.axis('equal')

   

    plt.tight_layout()

    plt.savefig('summary_results.png', dpi=300, bbox_inches='tight')

    plt.show()

   

    # Print summary table

    print("\nSummary Table:")

    print(f"{'K':<5} {'MAP Estimate (x, y)':<30} {'Error':<10}")

    print("-" * 50)

    for r in results:

        est_str = f"({r['map_estimate'][0]:.4f}, {r['map_estimate'][1]:.4f})"

        print(f"{r['K']:<5} {est_str:<30} {r['error']:.4f}")

   

    print("\n" + "=" * 70)

    print("Analysis Complete!")

    print("Plots saved: all_contours_combined.png, summary_results.png")

    print("=" * 70)



if __name__ == "__main__":

    main()

