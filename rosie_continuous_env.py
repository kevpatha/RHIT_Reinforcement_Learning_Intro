import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import time
from matplotlib.patches import Circle, Rectangle, Polygon, Ellipse

class RosieContinuousEnv:
    """
    A continuous 10x10 meter environment for Reinforcement Learning.
    The agent (Rosie the elephant) moves with acceleration and slides due to low friction.
    Accelerations are caused by Rosie blowing water out of her trunk.
    """
    def __init__(self):
        self.grid_size = 10.0  # 10x10 meters
        self.goal_pos = np.array([9.0, 9.0])  # Goal at (9, 9)
        self.goal_radius = 0.5
        self.holes = [
            np.array([2.0, 2.0]), np.array([4.0, 7.0]), np.array([8.0, 3.0]), 
            np.array([1.0, 6.0]), np.array([6.0, 1.0])
        ]
        self.hole_radius = 0.4
        
        # Action space: 0: None, 1: Up (y-), 2: Down (y+), 3: Left (x-), 4: Right (x+)
        self.action_space = {
            0: "NONE",
            1: "UP",
            2: "DOWN",
            3: "LEFT",
            4: "RIGHT"
        }
        
        # State: (x, y, vx, vy)
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        
        self.accel_mag = 2  # m/s^2
        self.friction = 0.5  # friction coefficient
        self.dt = 0.1         # Time step (seconds)
        
        self.reset()

    def reset(self):
        """
        Resets the environment. Random position, zero velocity.
        """
        max_attempts = 100
        for _ in range(max_attempts):
            self.state = np.array([
                np.random.uniform(0.5, self.grid_size - 0.5), # x
                np.random.uniform(0.5, self.grid_size - 0.5), # y
                0.0, # vx
                0.0  # vy
            ])
            
            # Check if starting in goal or hole
            is_term, _ = self._is_terminal(self.state[:2])
            if not is_term:
                break
        
        return self.state

    def _is_terminal(self, pos):
        # Goal check
        dist_to_goal = np.linalg.norm(pos - self.goal_pos)
        if dist_to_goal < self.goal_radius:
            return True, 100  # done, reward
        
        # Hole check
        for hole in self.holes:
            if np.linalg.norm(pos - hole) < self.hole_radius:
                return True, -100 # done, reward
        
        return False, 0

    def step(self, action):
        """
        Updates the environment based on the action.
        """
        x, y, vx, vy = self.state
        ax, ay = 0.0, 0.0
        
        if action == 1:  # UP (y decreases)
            ay = -self.accel_mag
        elif action == 2:  # DOWN (y increases)
            ay = self.accel_mag
        elif action == 3:  # LEFT (x decreases)
            ax = -self.accel_mag
        elif action == 4:  # RIGHT (x increases)
            ax = self.accel_mag
            
        # Update velocity: v = v + a*dt - friction*v
        vx += (ax - self.friction * np.sign(vx))*self.dt
        vy += (ay - self.friction * np.sign(vy))*self.dt
        
        # Update position: p = p + v*dt
        x += vx * self.dt
        y += vy * self.dt
        
        # Boundary conditions (bounce)
        if x < 0 or x > self.grid_size:
            vx = -vx * 0.5 # Lose some energy on bounce
            x = np.clip(x, 0, self.grid_size)
        if y < 0 or y > self.grid_size:
            vy = -vy * 0.5
            y = np.clip(y, 0, self.grid_size)
            
        self.state = np.array([x, y, vx, vy])
        
        done, reward = self._is_terminal(self.state[:2])
        if not done:
            reward = -1 # Step penalty
            
        return self.state, reward, done

    def run_inline(self, policy_fn=None, max_steps=200):
        """
        Visualizes an episode in a Jupyter Notebook.
        policy_fn is a function that takes state and returns an action.
        """
        state = self.reset()
        done = False
        steps = 0
        
        # Colors
        rose_red = "#800000"
        white = "#FFFFFF"
        black = "#000000"
        orange = "#FFA500"
        brown = "#8B4513"
        gray = "#808080"
        water_blue = "#ADD8E6"

        last_action = 0

        while not done and steps < max_steps:
            if policy_fn:
                action = policy_fn(state)
            else:
                # Random action for demo if no policy
                action = np.random.randint(0, 5)
                
            last_action = action
            state, reward, done = self.step(action)
            steps += 1

            if steps % 1 == 0 or done:
                clear_output(wait=True)
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_xlim(0, self.grid_size)
                ax.set_ylim(self.grid_size, 0) # y-down to match grid env
                ax.set_aspect('equal')
                ax.set_xticks(range(int(self.grid_size) + 1))
                ax.set_yticks(range(int(self.grid_size) + 1))
                ax.grid(True, color=rose_red, linewidth=0.5, alpha=0.3)
                ax.set_title(f"Step: {steps} | Pos: ({state[0]:.2f}, {state[1]:.2f}) | Vel: ({state[2]:.2f}, {state[3]:.2f})", color=rose_red)

                # Draw Goal (Bonfire)
                gx, gy = self.goal_pos
                ax.add_patch(Rectangle((gx - 0.3, gy + 0.1), 0.6, 0.1, color=brown))
                ax.add_patch(Rectangle((gx - 0.2, gy + 0.2), 0.4, 0.1, color=brown))
                flame = Polygon([
                    [gx - 0.2, gy + 0.1],
                    [gx, gy - 0.4],
                    [gx + 0.2, gy + 0.1]
                ], color=orange, ec=rose_red)
                ax.add_patch(flame)
                ax.text(gx, gy + 0.4, "BONFIRE", color=rose_red, ha='center', fontsize=8, weight='bold')

                # Draw Holes
                for hx, hy in self.holes:
                    hole = Circle((hx, hy), self.hole_radius, color=black, ec='white', linewidth=1.5)
                    ax.add_patch(hole)
                    # Inner depth for better visibility
                    ax.add_patch(Circle((hx, hy), self.hole_radius * 0.7, color='#1a1a1a'))
                    ax.text(hx, hy, "HOLE", color="white", ha='center', va='center', fontsize=8, weight='bold')

                # Determine Rosie's Orientation
                angle = 0 # Default facing right
                if last_action == 1: # Moving UP, face DOWN (90 deg)
                    angle = 90
                elif last_action == 2: # Moving DOWN, face UP (-90 deg)
                    angle = -90
                elif last_action == 3: # Moving LEFT, face RIGHT (0 deg)
                    angle = 0
                elif last_action == 4: # Moving RIGHT, face LEFT (180 deg)
                    angle = 180
                else:
                    # If no accel, face movement direction
                    if abs(state[2]) > 0.01 or abs(state[3]) > 0.01:
                        angle = np.degrees(np.arctan2(state[3], state[2]))
                    else:
                        angle = 0

                # Draw Rosie (Elephant)
                rx, ry = state[0], state[1]
                
                # Helper to rotate points around Rosie's center
                def rotate_point_local(px, py, cx, cy, deg):
                    rad = np.radians(deg)
                    nx = cx + (px - cx) * np.cos(rad) - (py - cy) * np.sin(rad)
                    ny = cy + (px - cx) * np.sin(rad) + (py - cy) * np.cos(rad)
                    return nx, ny

                # Ears (drawn behind head)
                for side in [-1, 1]:
                    ex, ey = rotate_point_local(rx + 0.3, ry + side*0.1, rx, ry, angle)
                    ear = Ellipse((ex, ey), 0.15, 0.2, angle=angle, color=gray, ec=black)
                    ax.add_patch(ear)

                # Body (Ellipse to allow rotation)
                body = Ellipse((rx, ry), 0.6, 0.4, angle=angle, color=gray, ec=black)
                ax.add_patch(body)
                
                # Head (at the "front" of the angle)
                hx, hy = rotate_point_local(rx + 0.3, ry, rx, ry, angle)
                head = Circle((hx, hy), 0.15, color=gray, ec=black)
                ax.add_patch(head)

                # Trunk
                tx_start, ty_start = rotate_point_local(rx + 0.4, ry, rx, ry, angle)
                
                # Water jet direction relative to trunk tip (opposite of motion)
                water_dx, water_dy = 0, 0
                if last_action == 1: # Moving UP, Water DOWN
                    water_dy = 1.0
                elif last_action == 2: # Moving DOWN, Water UP
                    water_dy = -1.0
                elif last_action == 3: # Moving LEFT, Water RIGHT
                    water_dx = 1.0
                elif last_action == 4: # Moving RIGHT, Water LEFT
                    water_dx = -1.0

                if last_action != 0:
                    # Water jet direction
                    tx_end, ty_end = rotate_point_local(rx + 0.4 + water_dx*0.3, ry + water_dy*0.3, rx, ry, angle)
                else:
                    tx_end, ty_end = rotate_point_local(rx + 0.5, ry + 0.1, rx, ry, angle)
                    
                ax.plot([tx_start, tx_end], [ty_start, ty_end], color=gray, linewidth=3)

                # Water Animation (if accelerating)
                if last_action != 0:
                    # Water origin at trunk tip
                    for _ in range(12):
                        wx = tx_end + water_dx * np.random.uniform(0.1, 0.8) + np.random.normal(0, 0.1)
                        wy = ty_end + water_dy * np.random.uniform(0.1, 0.8) + np.random.normal(0, 0.1)
                        ax.add_patch(Circle((wx, wy), 0.05, color=water_blue, alpha=0.6))
                
                # Eye
                ex, ey = rotate_point_local(rx + 0.35, ry - 0.05, rx, ry, angle)
                ax.add_patch(Circle((ex, ey), 0.02, color=black))

                display(fig)
                plt.close(fig)
                time.sleep(self.dt)

        if done:
            _, final_reward = self._is_terminal(state[:2])
            if final_reward > 0:
                print(f"Goal Reached in {steps} steps!")
            else:
                print(f"Fell into a hole after {steps} steps!")
        else:
            print("Max steps reached.")

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
        rx, ry, _, _ = self.state
        ax.add_patch(plt.Circle((rx, ry), 0.3, color='red'))
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.array(canvas.buffer_rgba())[..., :3] # Get RGB from RGBA buffer
        
        # Convert to grayscale and normalize
        gray = np.dot(rgba[..., :3], [0.2989, 0.5870, 0.1140])
        plt.close(fig)
        return gray / 255.0

    def run_gui(self, policy_fn=None):
        """
        Launches a separate window with a GUI to visualize the environment.
        If policy_fn is provided, it uses the policy to choose actions.
        Otherwise, allows the user to control Rosie with arrow keys.
        """
        root = tk.Tk()
        root.title("RHIT Continuous Rosie Environment")
        
        cell_size = 60
        width = int(self.grid_size * cell_size)
        height = int(self.grid_size * cell_size)
        
        canvas = tk.Canvas(root, width=width, height=height, bg="white")
        canvas.pack()

        # Colors
        rose_red = "#800000"
        white = "#FFFFFF"
        black = "#000000"
        orange = "#FFA500"
        brown = "#8B4513"
        gray = "#808080"
        water_blue = "#ADD8E6"

        self.current_action = 0
        self.last_accel_action = 0
        self.reset()
        self.gui_done = False
        self.gui_steps = 0

        def key_pressed(event):
            if policy_fn is not None: return # Ignore keys if policy is running
            if event.keysym == 'Up':
                self.current_action = 1
                self.last_accel_action = 1
            elif event.keysym == 'Down':
                self.current_action = 2
                self.last_accel_action = 2
            elif event.keysym == 'Left':
                self.current_action = 3
                self.last_accel_action = 3
            elif event.keysym == 'Right':
                self.current_action = 4
                self.last_accel_action = 4
            elif event.keysym == 'n':
                self.current_action = 0

        def key_released(event):
            if policy_fn is not None: return
            # When key is released, stop accelerating
            self.current_action = 0

        root.bind("<KeyPress>", key_pressed)
        root.bind("<KeyRelease>", key_released)

        def rotate_point(px, py, cx, cy, deg):
            rad = np.radians(deg)
            nx = cx + (px - cx) * np.cos(rad) - (py - cy) * np.sin(rad)
            ny = cy + (px - cx) * np.sin(rad) + (py - cy) * np.cos(rad)
            return nx, ny

        def draw():
            canvas.delete("all")
            
            # Draw grid lines
            for i in range(int(self.grid_size) + 1):
                canvas.create_line(i * cell_size, 0, i * cell_size, height, fill=rose_red, dash=(2, 4))
                canvas.create_line(0, i * cell_size, width, i * cell_size, fill=rose_red, dash=(2, 4))

            # Draw Goal (Bonfire)
            gx, gy = self.goal_pos
            gx_px, gy_px = gx * cell_size, gy * cell_size
            # Logs
            canvas.create_rectangle(gx_px - 0.3*cell_size, gy_px + 0.1*cell_size, gx_px + 0.3*cell_size, gy_px + 0.2*cell_size, fill=brown, outline="")
            canvas.create_rectangle(gx_px - 0.2*cell_size, gy_px + 0.2*cell_size, gx_px + 0.2*cell_size, gy_px + 0.3*cell_size, fill=brown, outline="")
            # Flame
            flame_pts = [
                gx_px - 0.2*cell_size, gy_px + 0.1*cell_size,
                gx_px, gy_px - 0.4*cell_size,
                gx_px + 0.2*cell_size, gy_px + 0.1*cell_size
            ]
            canvas.create_polygon(flame_pts, fill=orange, outline=rose_red)
            canvas.create_text(gx_px, gy_px + 0.4*cell_size, text="BONFIRE", fill=rose_red, font=("Arial", 8, "bold"))

            # Draw Holes
            for hx, hy in self.holes:
                hx_px, hy_px = hx * cell_size, hy * cell_size
                r_px = self.hole_radius * cell_size
                # Draw hole with white outline and inner depth
                canvas.create_oval(hx_px - r_px, hy_px - r_px, hx_px + r_px, hy_px + r_px, fill=black, outline="white", width=2)
                canvas.create_oval(hx_px - r_px*0.7, hy_px - r_px*0.7, hx_px + r_px*0.7, hy_px + r_px*0.7, fill="#1a1a1a", outline="")
                canvas.create_text(hx_px, hy_px, text="HOLE", fill="white", font=("Arial", 8, "bold"))

            # State
            rx, ry, vx, vy = self.state
            rx_px, ry_px = rx * cell_size, ry * cell_size

            # Orientation
            angle = 0
            if self.last_accel_action == 1: # Moving UP, face DOWN
                angle = 90
            elif self.last_accel_action == 2: # Moving DOWN, face UP
                angle = -90
            elif self.last_accel_action == 3: # Moving LEFT, face RIGHT
                angle = 0
            elif self.last_accel_action == 4: # Moving RIGHT, face LEFT
                angle = 180
            else:
                if abs(vx) > 0.01 or abs(vy) > 0.01:
                    angle = np.degrees(np.arctan2(vy, vx))
                else:
                    angle = 0
            
            # TKinter angle is different (clockwise, 0 is right). 
            # Actually, standard rotate_point works with standard math coords.
            # But y is inverted in GUI.
            
            # Helper for GUI rotation (handles y-inversion)
            def rotate_gui(px, py, cx, cy, deg):
                # Standard rotation then flip y for display? 
                # Better: just use the logic but remember y-down.
                rad = np.radians(deg)
                # For y-down, positive deg is clockwise.
                nx = cx + (px - cx) * np.cos(rad) - (py - cy) * np.sin(rad)
                ny = cy + (px - cx) * np.sin(rad) + (py - cy) * np.cos(rad)
                return nx, ny

            # Ears
            for side in [-1, 1]:
                ear_pts = []
                # Center of head in GUI: rx_px + 18, ry_px
                for d in range(0, 361, 30):
                    ex, ey = rotate_gui(rx_px + 18 + 4*np.cos(np.radians(d)), ry_px + side*8 + 6*np.sin(np.radians(d)), rx_px, ry_px, angle)
                    ear_pts.extend([ex, ey])
                canvas.create_polygon(ear_pts, fill="gray", outline="black")

            # Rosie Body
            body_rx, body_ry = 0.3 * cell_size, 0.2 * cell_size
            body_poly = []
            for d in range(0, 361, 20):
                ex_pt = rx_px + body_rx * np.cos(np.radians(d))
                ey_pt = ry_px + body_ry * np.sin(np.radians(d))
                nx, ny = rotate_gui(ex_pt, ey_pt, rx_px, ry_px, angle)
                body_poly.extend([nx, ny])
            canvas.create_polygon(body_poly, fill=gray, outline=black)

            # Head
            hx, hy = rotate_gui(rx_px + 18, ry_px, rx_px, ry_px, angle)
            hr = 9
            canvas.create_oval(hx-hr, hy-hr, hx+hr, hy+hr, fill="gray", outline="black")

            # Trunk
            tx_start, ty_start = rotate_gui(rx_px + 0.4*cell_size, ry_px, rx_px, ry_px, angle)
            
            water_dx, water_dy = 0, 0
            if self.current_action == 1: water_dy = 1.0 # Moving UP, Water DOWN
            elif self.current_action == 2: water_dy = -1.0 # Moving DOWN, Water UP
            elif self.current_action == 3: water_dx = 1.0 # Moving LEFT, Water RIGHT
            elif self.current_action == 4: water_dx = -1.0 # Moving RIGHT, Water LEFT

            if self.current_action != 0:
                # Trunk points in direction of water jet
                tx_end, ty_end = rotate_gui(rx_px + 0.4*cell_size + water_dx*0.2*cell_size, 
                                          ry_px + water_dy*0.2*cell_size, rx_px, ry_px, angle)
                # Water droplets
                for _ in range(10):
                    wx = tx_end + water_dx * np.random.uniform(5, 40) + np.random.normal(0, 5)
                    wy = ty_end + water_dy * np.random.uniform(5, 40) + np.random.normal(0, 5)
                    canvas.create_oval(wx-2, wy-2, wx+2, wy+2, fill=water_blue, outline="")
            else:
                tx_end, ty_end = rotate_gui(rx_px + 0.5*cell_size, ry_px + 0.1*cell_size, rx_px, ry_px, angle)

            canvas.create_line(tx_start, ty_start, tx_end, ty_end, fill=gray, width=3)

            # Eye
            ex, ey = rotate_gui(rx_px + 0.35*cell_size, ry_px - 0.05*cell_size, rx_px, ry_px, angle)
            er = 2
            canvas.create_oval(ex - er, ey - er, ex + er, ey + er, fill=black)

            # Info
            canvas.create_text(10, 10, anchor="nw", text=f"Steps: {self.gui_steps} | Pos: ({rx:.2f}, {ry:.2f})", fill=rose_red)

        def update():
            if not self.gui_done:
                if policy_fn is not None:
                    self.current_action = policy_fn(self.state)
                    if self.current_action != 0:
                        self.last_accel_action = self.current_action

                self.state, reward, done = self.step(self.current_action)
                self.gui_steps += 1
                draw()
                
                if done:
                    self.gui_done = True
                    _, final_reward = self._is_terminal(self.state[:2])
                    msg = "Goal Reached!" if final_reward > 0 else "Fell into Hole!"
                    canvas.create_text(width/2, height/2, text=msg, font=("Arial", 24, "bold"), fill=rose_red)
                else:
                    root.after(ms=int(self.dt * 1000), func=update)

        update()
        root.mainloop()

if __name__ == "__main__":
    # Test the environment
    env = RosieContinuousEnv()
    # env.run_inline(max_steps=50) # Commented out for GUI test
    env.run_gui()
