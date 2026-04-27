import numpy as np
import tkinter as tk

class GridWorldEnv:
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
            3: "RIGHT",
            4: "NONE"
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
        elif action == 4:  # NONE
            new_x, new_y = x, y
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

    def run_gui(self):
        """
        Launches a simple GUI using tkinter to visualize the environment.
        Allows the user to control the agent with arrow keys or 'n' for none.
        Features Rose-Hulman colors, an elephant agent, and a homecoming bonfire goal.
        """
        root = tk.Tk()
        root.title("RHIT GridWorld RL Environment")
        
        # RHIT Colors
        rose_red = "#800000"
        white = "#FFFFFF"
        black = "#000000"
        orange = "#FFA500" # For bonfire flames
        brown = "#8B4513"  # For bonfire logs
        gray = "#808080"   # For elephant body
        
        cell_size = 60
        canvas_size = self.grid_size * cell_size
        canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg=white)
        canvas.pack()

        def draw_grid():
            canvas.delete("all")
            # Draw grid lines and cells
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    x1, y1 = i * cell_size, j * cell_size
                    x2, y2 = x1 + cell_size, y1 + cell_size
                    
                    # RHIT scheme: White cells, Rose Red outlines
                    canvas.create_rectangle(x1, y1, x2, y2, fill=white, outline=rose_red)
                    
                    if (i, j) == self.goal_pos:
                        # Draw Bonfire (Goal)
                        # Logs
                        canvas.create_rectangle(x1 + 10, y2 - 15, x2 - 10, y2 - 5, fill=brown)
                        canvas.create_rectangle(x1 + 15, y2 - 25, x2 - 15, y2 - 15, fill=brown)
                        # Flames (Polygon)
                        canvas.create_polygon(
                            x1 + 15, y2 - 25,
                            x1 + 30, y1 + 5,
                            x2 - 15, y2 - 25,
                            fill=orange, outline=rose_red
                        )
                        canvas.create_text(x1 + cell_size/2, y2 - 5, text="BONFIRE", fill=rose_red, font=("Arial", 8, "bold"))
                    elif (i, j) in self.holes:
                        # Draw Holes (Black)
                        canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill=black)
                        canvas.create_text(x1 + cell_size/2, y1 + cell_size/2, text="HOLE", fill="white", font=("Arial", 8, "bold"))

            # Draw Agent (Elephant)
            ax, ay = self.agent_pos
            ax1, ay1 = ax * cell_size, ay * cell_size
            
            # Elephant body (Oval)
            canvas.create_oval(ax1 + 15, ay1 + 20, ax1 + 50, ay1 + 50, fill=gray, outline=black, tags="agent")
            # Elephant head (Oval)
            canvas.create_oval(ax1 + 35, ay1 + 15, ax1 + 55, ay1 + 35, fill=gray, outline=black, tags="agent")
            # Elephant trunk (Line/Polygon)
            canvas.create_line(ax1 + 50, ay1 + 30, ax1 + 60, ay1 + 45, width=4, fill=gray, tags="agent")
            # Elephant ears (Oval)
            canvas.create_oval(ax1 + 30, ay1 + 10, ax1 + 45, ay1 + 30, fill=gray, outline=black, tags="agent")
            # Eye
            canvas.create_oval(ax1 + 48, ay1 + 22, ax1 + 50, ay1 + 24, fill=black, tags="agent")

        def handle_keypress(event):
            action = None
            if event.keysym == "Up": action = 0
            elif event.keysym == "Down": action = 1
            elif event.keysym == "Left": action = 2
            elif event.keysym == "Right": action = 3
            elif event.keysym == "n": action = 4
            
            if action is not None:
                state, reward, done = self.step(action)
                print(f"Action {self.action_space[action]}: State {state}, Reward {reward}, Done {done}")
                draw_grid()
                if done:
                    print("Episode finished. Resetting...")
                    self.reset()
                    draw_grid()

        draw_grid()
        root.bind("<Key>", handle_keypress)
        print("Control the agent with arrow keys (Up, Down, Left, Right). Press 'n' for None action.")
        root.mainloop()

if __name__ == "__main__":
    # Quick verification
    env = GridWorldEnv()
    
    # Launch GUI
    env.run_gui()
