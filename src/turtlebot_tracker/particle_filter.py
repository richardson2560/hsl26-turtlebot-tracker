import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple

class SDEParticleFilter:
    """Tracks Turtlebot2 6-DOF state [x, y, z, phi, theta, psi, vx, vy, vz, wz] using stochastic differential equations."""
    
    def __init__(self, num_particles: int = 150, process_noise_std: float = 0.05):
        self.num_particles = num_particles
        self.dt = 0.1  # Default timestep (10 Hz)
        
        # State: [x, y, z, roll, pitch, yaw, vx, vy, vz, wz]
        self.particles = np.zeros((num_particles, 10))
        self.weights = np.ones(num_particles) / num_particles
        self.process_noise = process_noise_std

    def initialize(self, initial_pose_t: np.ndarray, initial_pose_R: np.ndarray):
        """Initializes particles around first measurement."""
        r_euler = R.from_matrix(initial_pose_R).as_euler('xyz')
        
        for i in range(self.num_particles):
            self.particles[i, 0:3] = initial_pose_t + np.random.normal(0, 0.05, 3)
            self.particles[i, 3:6] = r_euler + np.random.normal(0, 0.02, 3)
            self.particles[i, 6:10] = np.random.normal(0, 0.01, 4)  # Initial velocities
            
        self.weights.fill(1.0 / self.num_particles)

    def predict(self, dt: float):
        """Euler-Maruyama step for Stochastic Differential Equation: dX_t = f(X_t)dt + G dW_t."""
        self.dt = dt
        sqrt_dt = np.sqrt(dt)

        for i in range(self.num_particles):
            x, y, z, roll, pitch, yaw, vx, vy, vz, wz = self.particles[i]
            
            # Kinematic Drift f(X_t)
            dx = vx * dt
            dy = vy * dt
            dz = vz * dt
            droll = 0.0
            dpitch = 0.0
            dyaw = wz * dt
            
            # Wiener process noise dW_t
            noise = np.random.normal(0, self.process_noise, 10) * sqrt_dt
            
            self.particles[i, 0] += dx + noise[0]
            self.particles[i, 1] += dy + noise[1]
            self.particles[i, 2] += dz + noise[2]
            self.particles[i, 3] += droll + noise[3]
            self.particles[i, 4] += dpitch + noise[4]
            self.particles[i, 5] += dyaw + noise[5]
            
            # Velocity random walk
            self.particles[i, 6:10] += noise[6:10]

    def update(self, measured_t: np.ndarray, measured_R: np.ndarray, ot_cost: float):
        """Updates particle weights based on transport cost and likelihood."""
        meas_euler = R.from_matrix(measured_R).as_euler('xyz')

        for i in range(self.num_particles):
            p_t = self.particles[i, 0:3]
            p_euler = self.particles[i, 3:6]

            pos_err_sq = np.sum((p_t - measured_t) ** 2)
            rot_err_sq = np.sum((p_euler - meas_euler) ** 2)

            # Likelihood evaluation
            likelihood = np.exp(-pos_err_sq / (2 * 0.1**2) - rot_err_sq / (2 * 0.1**2) - ot_cost / 1.0)
            self.weights[i] *= likelihood + 1e-12

        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights.fill(1.0 / self.num_particles)

        # Resample if effective sample size drops
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.num_particles / 2.0:
            self._systematic_resample()

    def _systematic_resample(self):
        """Low-variance systematic resampling."""
        indices = np.zeros(self.num_particles, dtype=int)
        cdf = np.cumsum(self.weights)
        u1 = np.random.uniform(0, 1.0 / self.num_particles)

        j = 0
        for i in range(self.num_particles):
            u = u1 + i / self.num_particles
            while u > cdf[j] and j < self.num_particles - 1:
                j += 1
            indices[i] = j

        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def get_estimated_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns mean estimated translation vector [3] and rotation matrix [3, 3]."""
        mean_state = np.average(self.particles, weights=self.weights, axis=0)
        mean_t = mean_state[0:3]
        mean_euler = mean_state[3:6]
        mean_R = R.from_euler('xyz', mean_euler).as_matrix()
        return mean_t, mean_R