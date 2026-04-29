import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import time

class RosieGridEnv:
    """
    A 10x10 Grid World Markov Decision Process Environment for Reinforcement Learning.
    The agent starts at a random position and must reach the target goal.
    There are five holes on the grid that end the episode with a negative reward.
    """
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.goal_pos = (grid_size - 1, grid_size - 1)  # Target: bottom-right corner
        self.holes = [
            (2, 2), (4, 7), (8, 3), (1, 6), (6, 1)
        ]
        self.action_space = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT"
        }
        self.reset()

    def reset(self):
        """
        Resets the environment to a randomized starting position.
        The starting position will not be the goal or a hole.
        """
        while True:
            start_x = np.random.randint(0, self.grid_size)
            start_y = np.random.randint(0, self.grid_size)
            self.agent_pos = (start_x, start_y)
            
            if self.agent_pos != self.goal_pos and self.agent_pos not in self.holes:
                break
        
        return self.agent_pos

    def step(self, action):
        """
        Executes a step in the environment based on the provided action.
        
        Actions:
        0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: NONE
        
        Returns:
        new_state (tuple): The agent's new (x, y) position.
        reward (int): The reward received after the step.
        done (bool): Whether the episode has ended.
        """
        x, y = self.agent_pos
        
        # Determine new position based on action
        if action == 0:  # UP
            new_x, new_y = x, max(0, y - 1)
        elif action == 1:  # DOWN
            new_x, new_y = x, min(self.grid_size - 1, y + 1)
        elif action == 2:  # LEFT
            new_x, new_y = max(0, x - 1), y
        elif action == 3:  # RIGHT
            new_x, new_y = min(self.grid_size - 1, x + 1), y
        else:
            raise ValueError(f"Invalid action: {action}. Must be 0, 1, 2, 3, or 4.")

        self.agent_pos = (new_x, new_y)
        
        # Determine reward and if episode ends
        if self.agent_pos == self.goal_pos:
            reward = 1
            done = True
        elif self.agent_pos in self.holes:
            reward = -10
            done = True
        else:
            reward = -1
            done = False
            
        return self.agent_pos, reward, done

    def run_inline(self, q_table=None, max_steps=100):
        """
        Runs an episode using greedy actions from the q_table and displays it inline in a Jupyter Notebook.
        """
        if q_table is None:
            print("Please provide a Q-table for inline visualization.")
            return

        self.reset()
        done = False
        steps = 0

        # RHIT Colors
        rose_red = "#800000"
        white = "#FFFFFF"
        black = "#000000"
        orange = "#FFA500" # For bonfire flames
        brown = "#8B4513"  # For bonfire logs
        gray = "#808080"   # For elephant body

        while not done and steps < max_steps:
            # Select greedy action
            action = q_table[self.agent_pos[0], self.agent_pos[1]].argmax()
            state, reward, done = self.step(action)
            steps += 1

            # Only visualize periodically or at the end to avoid "hanging" in some environments
            # but keep the interactive feel if possible. 
            # Given the user's issue, we'll ensure the last frame is always shown.
            if steps % 1 == 0 or done:
                clear_output(wait=True)
                
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.set_xlim(0, self.grid_size)
                ax.set_ylim(0, self.grid_size)
                ax.set_aspect('equal')
                ax.set_xticks(range(self.grid_size + 1))
                ax.set_yticks(range(self.grid_size + 1))
                ax.grid(True, color=rose_red, linewidth=1)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_title(f"Step: {steps} | Pos: {self.agent_pos}", color=rose_red)

                # Draw Cells
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        # Coordinate conversion: j in our env is y-down, but plt y is y-up
                        plot_y = self.grid_size - 1 - j
                        
                        if (i, j) == self.goal_pos:
                            # Draw Bonfire (Goal)
                            # Logs
                            ax.add_patch(plt.Rectangle((i + 0.2, plot_y + 0.1), 0.6, 0.1, color=brown))
                            ax.add_patch(plt.Rectangle((i + 0.3, plot_y + 0.2), 0.4, 0.1, color=brown))
                            # Flames
                            flame = plt.Polygon([
                                [i + 0.3, plot_y + 0.3],
                                [i + 0.5, plot_y + 0.8],
                                [i + 0.7, plot_y + 0.3]
                            ], color=orange, ec=rose_red)
                            ax.add_patch(flame)
                            ax.text(i + 0.5, plot_y + 0.05, "BONFIRE", color=rose_red, ha='center', fontsize=8, weight='bold')
                        elif (i, j) in self.holes:
                            # Draw Hole
                            hole = plt.Circle((i + 0.5, plot_y + 0.5), 0.4, color=black)
                            ax.add_patch(hole)
                            ax.text(i + 0.5, plot_y + 0.5, "HOLE", color="white", ha='center', va='center', fontsize=8, weight='bold')

                # Draw Agent (Elephant)
                ax_agent, ay_agent = self.agent_pos
                aplot_y = self.grid_size - 1 - ay_agent
                
                # Body
                ax.add_patch(plt.Circle((ax_agent + 0.5, aplot_y + 0.5), 0.3, color=gray, ec=black))
                # Head
                ax.add_patch(plt.Circle((ax_agent + 0.7, aplot_y + 0.6), 0.15, color=gray, ec=black))
                # Ear
                ax.add_patch(plt.Circle((ax_agent + 0.6, aplot_y + 0.65), 0.1, color=gray, ec=black))
                # Trunk
                ax.plot([ax_agent + 0.8, ax_agent + 0.9], [aplot_y + 0.55, aplot_y + 0.4], color=gray, linewidth=3)
                # Eye
                ax.add_patch(plt.Circle((ax_agent + 0.75, aplot_y + 0.65), 0.02, color=black))

                display(fig)
                plt.close(fig)
                time.sleep(0.1) # Reduced sleep for smoother experience

        if done:
            print(f"Goal Reached in {steps} steps!")
        else:
            print("Max steps reached.")

    def plot_q_table_progression(self, q_snapshots, episodes_list):
        """
        Plots a grid of heatmaps representing the Q-values for each action over time.
        
        Args:
            q_snapshots (list): List of Q-table arrays captured during training.
            episodes_list (list): List of episode numbers corresponding to each snapshot.
        """
        num_snapshots = len(q_snapshots)
        num_actions = len(self.action_space)
        
        fig, axes = plt.subplots(num_snapshots, num_actions, figsize=(4 * num_actions, 4 * num_snapshots))
        
        # Ensure axes is 2D even if only one snapshot is provided
        if num_snapshots == 1:
            axes = np.expand_dims(axes, axis=0)
            
        action_names = [self.action_space[i] for i in range(num_actions)]
        
        # RHIT colors for colormap - using a gradient from white to rose red
        from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
        rose_red = "#800000"
        white = "#FFFFFF"
        cmap = LinearSegmentedColormap.from_list("RHIT_Cmap", [white, rose_red])

        # Calculate global min and max across all snapshots to keep the color scale consistent
        all_min = min(q.min() for q in q_snapshots)
        all_max = max(q.max() for q in q_snapshots)

        for i, (q_table, ep) in enumerate(zip(q_snapshots, episodes_list)):
            for a in range(num_actions):
                ax = axes[i, a]
                # In our env, state is (x, y). Q_table is [x, y, action]
                # For plotting with imshow, we want rows to be Y and columns to be X.
                # So we transpose the x,y slice and flip it if necessary to match the grid visualization
                # plt.imshow(data) expects data[row, col] -> data[y, x]
                # Our Q_table[x, y] needs to be plotted as Q_table[col, row]
                data = q_table[:, :, a].T
                
                # Use SymLogNorm to make differences more visible across a wide range of values
                # linthresh=1.0 means values between -1 and 1 are linear, others are logarithmic
                # Use global min/max for consistent scaling
                norm = SymLogNorm(linthresh=1.0, linscale=1.0, vmin=all_min, vmax=all_max, base=10)
                
                im = ax.imshow(data, cmap=cmap, origin='upper', norm=norm)
                
                # Add colorbar only to the last column to save space but provide scale
                if a == num_actions - 1:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Add goal and holes for context
                # Goal
                gx, gy = self.goal_pos
                ax.text(gx, gy, "★", color="gold", ha='center', va='center', fontsize=12, weight='bold')
                
                # Holes
                for (hx, hy) in self.holes:
                    ax.text(hx, hy, "O", color="black", ha='center', va='center', fontsize=10, weight='bold')

                if i == 0:
                    ax.set_title(f"Action: {action_names[a]}", fontsize=14, weight='bold', color=rose_red)
                
                if a == 0:
                    ax.set_ylabel(f"Episode {ep}", fontsize=14, weight='bold', color=rose_red)
                
                # Add grid lines
                ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
                ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        plt.show()

    def plot_policy(self, q_table):
        """
        Plots a single grid with arrows representing the greedy policy (highest Q-value) at each state.
        
        Args:
            q_table (numpy.ndarray): The Q-table representing the learned policy.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5) # Invert y-axis to match (0,0) at top-left
        ax.set_aspect('equal')
        
        # RHIT Colors
        rose_red = "#800000"
        white = "#FFFFFF"
        
        # Draw grid
        ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.grid(which='minor', color=rose_red, linestyle='-', linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Greedy Policy (Arrows represent best action)", fontsize=16, weight='bold', color=rose_red)

        # Mapping of actions to (dx, dy) for arrows
        # 0: UP (0, -1), 1: DOWN (0, 1), 2: LEFT (-1, 0), 3: RIGHT (1, 0)
        action_vectors = {
            0: (0, -0.3),
            1: (0, 0.3),
            2: (-0.3, 0),
            3: (0.3, 0)
        }

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) == self.goal_pos:
                    ax.text(x, y, "★", color="gold", ha='center', va='center', fontsize=20, weight='bold')
                    ax.text(x, y+0.4, "GOAL", color=rose_red, ha='center', va='top', fontsize=8, weight='bold')
                    continue
                
                if (x, y) in self.holes:
                    ax.text(x, y, "O", color="black", ha='center', va='center', fontsize=15, weight='bold')
                    ax.text(x, y+0.4, "HOLE", color="black", ha='center', va='top', fontsize=8, weight='bold')
                    continue
                
                # Find the best action
                q_values = q_table[x, y]
                # If all Q-values are equal (e.g., all 0), don't draw an arrow to avoid noise
                if np.all(q_values == q_values[0]):
                    continue
                
                best_action = q_values.argmax()
                dx, dy = action_vectors[best_action]
                
                # Draw arrow
                ax.arrow(x - dx*0.5, y - dy*0.5, dx, dy, 
                         head_width=0.2, head_length=0.2, fc=rose_red, ec=rose_red)

        plt.tight_layout()
        plt.show()

    def run_gui(self, q_table=None):
        """
        Launches a simple GUI using tkinter to visualize the environment.
        If q_table is provided, it runs an episode using greedy actions from the table.
        Otherwise, allows the user to control the agent with arrow keys.
        Press R or click Reset to start a new episode at any time.
        """
        root = tk.Tk()
        root.title("RHIT GridWorld RL Environment")

        # RHIT Colors
        rose_red = "#800000"
        white = "#FFFFFF"
        black = "#000000"
        orange = "#FFA500"
        brown = "#8B4513"
        gray = "#808080"

        cell_size = 60
        canvas_size = self.grid_size * cell_size
        canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg=white)
        canvas.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X)
        reset_btn = tk.Button(btn_frame, text="Reset (R)", font=("Arial", 12), bg=rose_red, fg=white)
        reset_btn.pack(pady=4)

        self._gui_done = False

        def draw_grid():
            canvas.delete("all")
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    x1, y1 = i * cell_size, j * cell_size
                    x2, y2 = x1 + cell_size, y1 + cell_size
                    canvas.create_rectangle(x1, y1, x2, y2, fill=white, outline=rose_red)
                    if (i, j) == self.goal_pos:
                        canvas.create_rectangle(x1 + 10, y2 - 15, x2 - 10, y2 - 5, fill=brown)
                        canvas.create_rectangle(x1 + 15, y2 - 25, x2 - 15, y2 - 15, fill=brown)
                        canvas.create_polygon(
                            x1 + 15, y2 - 25, x1 + 30, y1 + 5, x2 - 15, y2 - 25,
                            fill=orange, outline=rose_red
                        )
                        canvas.create_text(x1 + cell_size/2, y2 - 5, text="BONFIRE", fill=rose_red, font=("Arial", 8, "bold"))
                    elif (i, j) in self.holes:
                        canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill=black)
                        canvas.create_text(x1 + cell_size/2, y1 + cell_size/2, text="HOLE", fill="white", font=("Arial", 8, "bold"))

            ax, ay = self.agent_pos
            ax1, ay1 = ax * cell_size, ay * cell_size
            canvas.create_oval(ax1 + 15, ay1 + 20, ax1 + 50, ay1 + 50, fill=gray, outline=black, tags="agent")
            canvas.create_oval(ax1 + 35, ay1 + 15, ax1 + 55, ay1 + 35, fill=gray, outline=black, tags="agent")
            canvas.create_line(ax1 + 50, ay1 + 30, ax1 + 60, ay1 + 45, width=4, fill=gray, tags="agent")
            canvas.create_oval(ax1 + 30, ay1 + 10, ax1 + 45, ay1 + 30, fill=gray, outline=black, tags="agent")
            canvas.create_oval(ax1 + 48, ay1 + 22, ax1 + 50, ay1 + 24, fill=black, tags="agent")

        def show_done_message(success):
            msg = "Goal Reached!" if success else "Fell in a Hole!"
            canvas.create_text(canvas_size/2, canvas_size/2, text=msg,
                                font=("Arial", 22, "bold"), fill=rose_red)
            canvas.create_text(canvas_size/2, canvas_size/2 + 36,
                                text="Press R or click Reset to play again",
                                font=("Arial", 13), fill=rose_red)

        def do_reset():
            self._gui_done = False
            self.reset()
            draw_grid()
            if q_table is not None:
                root.after(500, run_automated_step)

        reset_btn.config(command=do_reset)

        def handle_keypress(event):
            if event.keysym in ('r', 'R'):
                do_reset()
                return
            if q_table is not None:
                return
            action = None
            if event.keysym == "Up": action = 0
            elif event.keysym == "Down": action = 1
            elif event.keysym == "Left": action = 2
            elif event.keysym == "Right": action = 3
            if action is not None and not self._gui_done:
                state, reward, done = self.step(action)
                draw_grid()
                if done:
                    self._gui_done = True
                    show_done_message(self.agent_pos == self.goal_pos)

        def run_automated_step():
            if self._gui_done or q_table is None:
                return
            action = q_table[self.agent_pos[0], self.agent_pos[1]].argmax()
            _, reward, done = self.step(action)
            draw_grid()
            if done:
                self._gui_done = True
                show_done_message(self.agent_pos == self.goal_pos)
            else:
                root.after(500, run_automated_step)

        root.bind("<Key>", handle_keypress)
        self.reset()
        draw_grid()
        if q_table is not None:
            root.after(1000, run_automated_step)

        root.mainloop()

if __name__ == "__main__":
    # Quick verification
    env = RosieGridEnv()
    
    # Launch GUI
    env.run_gui()
