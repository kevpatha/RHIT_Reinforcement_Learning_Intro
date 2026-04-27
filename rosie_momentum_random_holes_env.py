import numpy as np
from rosie_momentum_env import RosieMomentumEnv

class RosieMomentumRandomHolesEnv(RosieMomentumEnv):
    """
    An extension of RosieMomentumEnv where hole positions are randomized
    each episode, and the state includes distances to the nearest hole.
    """
    def __init__(self, num_holes=5):
        self.num_holes = num_holes
        self.holes = [np.array([0.0, 0.0]) for _ in range(self.num_holes)]
        super().__init__()
        
    def _get_nearest_hole_dist(self, pos):
        """
        Calculates the x and y distances to the nearest hole.
        """
        min_dist_sq = float('inf')
        nearest_hole = self.holes[0]
        
        for hole in self.holes:
            dist_sq = np.sum((pos - hole)**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_hole = hole
        
        # Return relative vector from Rosie to the nearest hole
        return nearest_hole - pos

    def reset(self):
        """
        Resets the environment with randomized hole positions.
        """
        # Randomize hole positions
        self.holes = []
        for _ in range(self.num_holes):
            while True:
                hole = np.random.uniform(1.0, self.grid_size - 1.0, size=2)
                # Ensure holes aren't too close to the start (typically near 0,0) or goal
                if np.linalg.norm(hole - self.goal_pos) > 1.5 and np.linalg.norm(hole - np.array([0.0, 0.0])) > 1.5:
                    self.holes.append(hole)
                    break
        
        # Call parent reset to handle Rosie's position and initial state
        # Parent reset sets self.state to [x, y, 0, 0, water]
        super().reset()
        
        # Augment state with nearest hole distances
        rel_hole = self._get_nearest_hole_dist(self.state[:2])
        self.full_state = np.concatenate([self.state, rel_hole])
        
        return self.full_state

    def step(self, action):
        """
        Updates the environment and returns the augmented state.
        """
        # Parent step updates self.state and returns (next_state, reward, done)
        # where next_state is the 5D state.
        next_state_5d, reward, done = super().step(action)
        
        # Update augmented state
        rel_hole = self._get_nearest_hole_dist(next_state_5d[:2])
        self.full_state = np.concatenate([next_state_5d, rel_hole])
        
        return self.full_state, reward, done

    def get_policy_state(self):
        return self.full_state

    def get_image_observation(self, size=(64, 64)):
        """
        Renders the environment to a small grayscale image for CNN input.
        Returns a numpy array of shape (size[0], size[1]).
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        fig, ax = plt.subplots(figsize=(2, 2), dpi=size[0]/2)
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(self.grid_size, 0) # Flip y-axis to match coordinate system
        ax.set_aspect('equal')
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Draw Goal
        ax.add_patch(plt.Circle(self.goal_pos, 0.4, color='green'))
        
        # Draw Holes
        for h_pos in self.holes:
            ax.add_patch(plt.Circle(h_pos, self.hole_radius, color='black', ec='white', lw=1))
            
        # Draw Rosie (as a simple dot for image input efficiency)
        rx, ry = self.state[:2]
        ax.add_patch(plt.Circle((rx, ry), 0.3, color='red'))
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.array(canvas.buffer_rgba())[..., :3] # Get RGB from RGBA buffer
        
        # Convert to grayscale and normalize
        gray = np.dot(rgba[..., :3], [0.2989, 0.5870, 0.1140])
        plt.close(fig)
        return gray / 255.0
