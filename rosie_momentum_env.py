import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import time
from matplotlib.patches import Circle, Rectangle, Polygon, Ellipse

class RosieMomentumEnv:
    """
    A continuous 10x10 meter environment for Reinforcement Learning.
    The agent (Rosie the elephant) moves using momentum transfer from water jet.
    Rosie has limited water, and her mass decreases as she sprays it.
    """
    def __init__(self):
        self.grid_size = 10.0  # 10x10 meters
        self.goal_pos = np.array([9.0, 9.0])
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
        
        # Physics Parameters
        self.baseline_mass = 10.0    # Rosie's mass when empty
        self.initial_water = 25.0    # Initial water mass
        self.v_water_rel = 10.0     # Constant relative velocity of water leaving the trunk (m/s)
        self.water_burn_rate = 10  # Mass of water sprayed per action step (kg/s)
        self.friction_coeff = 0.05  # Sliding friction coefficient (mu)
        self.g = 9.81               # Gravity
        self.dt = 0.1               # Time step (seconds)
        
        # State: (x, y, vx, vy, current_water_mass)
        self.state = np.array([0.0, 0.0, 0.0, 0.0, self.initial_water])
        
        self.reset()

    def reset(self):
        """
        Resets the environment. Random position, zero velocity, full water.
        """
        max_attempts = 100
        for _ in range(max_attempts):
            x = np.random.uniform(0.5, self.grid_size - 0.5)
            y = np.random.uniform(0.5, self.grid_size - 0.5)
            self.state = np.array([x, y, 0.0, 0.0, self.initial_water])
            
            # Check if starting in goal or hole
            is_term, _ = self._is_terminal(self.state[:2], self.state[4])
            if not is_term:
                break
        
        return self.state

    def _is_terminal(self, pos, water_mass):
        # Goal check
        dist_to_goal = np.linalg.norm(pos - self.goal_pos)
        if dist_to_goal < self.goal_radius:
            # Additional positive reward proportional to the mass of water leftover
            water_bonus = water_mass * 20.0 # Scaling factor for bonus
            return True, 100 + water_bonus
        
        # Hole check
        for hole in self.holes:
            if np.linalg.norm(pos - hole) < self.hole_radius:
                return True, -100
        
        # Out of water and stopped check
        # Rosie is "stopped" if velocity is very low
        vx, vy = self.state[2], self.state[3]
        if water_mass <= 0 and abs(vx) < 0.01 and abs(vy) < 0.01:
            return True, 0 # Equivalent to completed episode without reaching goal (terminal reward 0)
        
        return False, 0

    def step(self, action):
        """
        Updates the environment based on momentum transfer.
        """
        x, y, vx, vy, water_mass = self.state
        
        m_old = self.baseline_mass + water_mass
        dm = 0.0
        
        # If we have water, we can use it to propel ourselves
        if action != 0 and water_mass > 0:
            dm = min(self.water_burn_rate * self.dt, water_mass)
            water_mass -= dm
            
            # Momentum transfer: dv = - (dm / m_new) * v_water_rel
            # Relative velocity of water is in the OPPOSITE direction of the movement action
            # 1: UP (y-), water sprays DOWN (y+) -> v_rel_y = v_water_rel
            # 2: DOWN (y+), water sprays UP (y-) -> v_rel_y = -v_water_rel
            # 3: LEFT (x-), water sprays RIGHT (x+) -> v_rel_x = v_water_rel
            # 4: RIGHT (x+), water sprays LEFT (x-) -> v_rel_x = -v_water_rel
            
            m_new = self.baseline_mass + water_mass
            
            dvx, dvy = 0.0, 0.0
            if action == 1: # UP
                dvy = - (dm / m_new) * (-self.v_water_rel) # wait, v_rel is relative to trunk.
                # If we want to move UP (dvy < 0), we spray water DOWN (v_rel_y > 0 relative to trunk)
                # Change in velocity: v_new = (m_old*v_old - dm*v_water_abs) / m_new
                # v_water_abs = v_old + v_water_rel
                # v_new = (m_old*v_old - dm*(v_old + v_water_rel)) / m_new
                # v_new = ((m_old - dm)*v_old - dm*v_water_rel) / m_new
                # v_new = v_old - (dm/m_new) * v_water_rel
                
                # Let's define v_water_rel_vec as the velocity of water relative to Rosie.
                # To move UP (y decreases), water must go DOWN (y increases relative to Rosie).
                # Action 1 (UP): v_water_rel_vec = [0, v_water_rel]
                # Action 2 (DOWN): v_water_rel_vec = [0, -v_water_rel]
                # Action 3 (LEFT): v_water_rel_vec = [v_water_rel, 0]
                # Action 4 (RIGHT): v_water_rel_vec = [-v_water_rel, 0]
                
            v_rel_x, v_rel_y = 0.0, 0.0
            if action == 1: v_rel_y = self.v_water_rel
            elif action == 2: v_rel_y = -self.v_water_rel
            elif action == 3: v_rel_x = self.v_water_rel
            elif action == 4: v_rel_x = -self.v_water_rel
            
            vx -= (dm / m_new) * v_rel_x
            vy -= (dm / m_new) * v_rel_y
        
        # Friction
        # F_f = mu * m * g
        # a_f = F_f / m = mu * g (independent of mass for simple sliding friction)
        # BUT the user says: "this change in mass will also alter the impeding force from friction"
        # If F_f = mu * m * g, then F_f indeed changes with mass.
        # However, a_f = mu * g is constant.
        # To make it interesting and follow the prompt's hint, maybe they meant air resistance or something where a_f = F / m?
        # Let's stick to standard physics unless specified: a_f = mu * g.
        # If they meant F is constant, a_f would increase as mass decreases.
        
        m_curr = self.baseline_mass + water_mass
        accel_f = self.friction_coeff * self.g # This is 0.05 * 9.81 approx 0.5 m/s^2
        
        # Apply friction deceleration to velocity
        speed = np.sqrt(vx**2 + vy**2)
        if speed > 0:
            new_speed = max(0, speed - accel_f * self.dt)
            ratio = new_speed / speed
            vx *= ratio
            vy *= ratio
            
        # Update position
        x += vx * self.dt
        y += vy * self.dt
        
        # Boundary conditions (bounce)
        if x < 0 or x > self.grid_size:
            vx = -vx * 0.5
            x = np.clip(x, 0, self.grid_size)
        if y < 0 or y > self.grid_size:
            vy = -vy * 0.5
            y = np.clip(y, 0, self.grid_size)
            
        self.state = np.array([x, y, vx, vy, water_mass])
        
        done, reward = self._is_terminal(self.state[:2], water_mass)
        if not done:
            reward = -1 # Step penalty
            
        return self.state, reward, done

    def run_inline(self, policy_fn=None, max_steps=400):
        state = self.reset()
        done = False
        steps = 0
        last_action = 0
        
        # Colors
        rose_red = "#800000"
        black = "#000000"
        orange = "#FFA500"
        brown = "#8B4513"
        gray = "#808080"
        water_blue = "#ADD8E6"

        while not done and steps < max_steps:
            action = policy_fn(state) if policy_fn else np.random.randint(0, 5)
            last_action = action
            state, reward, done = self.step(action)
            steps += 1

            if steps % 1 == 0 or done:
                clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_xlim(0, self.grid_size)
                ax.set_ylim(self.grid_size, 0)
                ax.set_aspect('equal')
                ax.grid(True, color=rose_red, linewidth=0.5, alpha=0.3)
                
                # Display stats including water
                ax.set_title(f"Step: {steps} | Water: {state[4]:.2f} | Pos: ({state[0]:.2f}, {state[1]:.2f})", color=rose_red)

                # Draw Goal
                gx, gy = self.goal_pos
                ax.add_patch(Rectangle((gx - 0.3, gy + 0.1), 0.6, 0.1, color=brown))
                flame = Polygon([[gx - 0.2, gy + 0.1], [gx, gy - 0.4], [gx + 0.2, gy + 0.1]], color=orange, ec=rose_red)
                ax.add_patch(flame)

                # Draw Holes
                for hx, hy in self.holes:
                    ax.add_patch(Circle((hx, hy), self.hole_radius, color=black, ec='white', linewidth=1.5))
                    # Add a smaller inner circle to give a "depth" look
                    ax.add_patch(Circle((hx, hy), self.hole_radius * 0.7, color='#1a1a1a'))

                # Rosie Orientation (Facing opposite to jet)
                angle = 0
                if last_action == 1: angle = 90
                elif last_action == 2: angle = -90
                elif last_action == 3: angle = 0
                elif last_action == 4: angle = 180
                else:
                    if abs(state[2]) > 0.01 or abs(state[3]) > 0.01:
                        angle = np.degrees(np.arctan2(state[3], state[2]))
                    else: angle = 0

                # Draw Rosie
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

                # Body
                body = Ellipse((rx, ry), 0.6, 0.4, angle=angle, color=gray, ec=black)
                ax.add_patch(body)

                # Legs
                for lx_off, ly_off in [(-0.15, -0.15), (0.15, -0.15), (-0.15, 0.15), (0.15, 0.15)]:
                    lx, ly = rotate_point_local(rx + lx_off, ry + ly_off, rx, ry, angle)
                    leg = Rectangle((lx-0.05, ly-0.05), 0.1, 0.1, color=gray, ec=black)
                    ax.add_patch(leg)

                # Head
                hx, hy = rotate_point_local(rx + 0.3, ry, rx, ry, angle)
                head = Circle((hx, hy), 0.15, color=gray, ec=black)
                ax.add_patch(head)

                # Trunk and Water
                tx_start, ty_start = rotate_point_local(rx + 0.4, ry, rx, ry, angle)
                water_dx, water_dy = 0, 0
                if last_action == 1: water_dy = 1.0
                elif last_action == 2: water_dy = -1.0
                elif last_action == 3: water_dx = 1.0
                elif last_action == 4: water_dx = -1.0

                if last_action != 0 and state[4] > 0:
                    tx_end, ty_end = rotate_point_local(rx + 0.4 + water_dx*0.3, ry + water_dy*0.3, rx, ry, angle)
                    for _ in range(8):
                        wx = tx_end + water_dx * np.random.uniform(0.1, 0.6) + np.random.normal(0, 0.05)
                        wy = ty_end + water_dy * np.random.uniform(0.1, 0.6) + np.random.normal(0, 0.05)
                        ax.add_patch(Circle((wx, wy), 0.04, color=water_blue, alpha=0.6))
                else:
                    tx_end, ty_end = rotate_point_local(rx + 0.5, ry + 0.1, rx, ry, angle)
                ax.plot([tx_start, tx_end], [ty_start, ty_end], color=gray, linewidth=3)

                # Eye
                ex, ey = rotate_point_local(rx + 0.35, ry - 0.05, rx, ry, angle)
                ax.add_patch(Circle((ex, ey), 0.02, color=black))

                display(fig)
                plt.close(fig)
                time.sleep(self.dt)

    def get_policy_state(self):
        return self.state

    def run_gui(self, policy_fn=None):
        root = tk.Tk()
        root.title("Rosie Momentum Environment")
        cell_size = 60
        width, height = int(self.grid_size * cell_size), int(self.grid_size * cell_size)
        canvas = tk.Canvas(root, width=width, height=height, bg="white")
        canvas.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X)
        reset_btn = tk.Button(btn_frame, text="Reset (R)", font=("Arial", 12), bg="#800000", fg="white")
        reset_btn.pack(pady=4)

        self.current_action = 0
        self.last_accel_action = 0
        self.reset()
        self.gui_done = False
        self.gui_steps = 0

        def do_reset():
            was_done = self.gui_done
            self.current_action = 0
            self.last_accel_action = 0
            self.reset()
            self.gui_done = False
            self.gui_steps = 0
            if was_done:
                root.after(int(self.dt * 1000), update)

        reset_btn.config(command=do_reset)

        def key_pressed(event):
            if event.keysym in ('r', 'R'):
                do_reset()
                return
            if policy_fn: return
            if event.keysym == 'Up': self.current_action = 1; self.last_accel_action = 1
            elif event.keysym == 'Down': self.current_action = 2; self.last_accel_action = 2
            elif event.keysym == 'Left': self.current_action = 3; self.last_accel_action = 3
            elif event.keysym == 'Right': self.current_action = 4; self.last_accel_action = 4
        def key_released(event):
            if not policy_fn: self.current_action = 0
        root.bind("<KeyPress>", key_pressed)
        root.bind("<KeyRelease>", key_released)

        def rotate_gui(px, py, cx, cy, deg):
            rad = np.radians(deg)
            return cx + (px - cx) * np.cos(rad) - (py - cy) * np.sin(rad), \
                   cy + (px - cx) * np.sin(rad) + (py - cy) * np.cos(rad)

        def update():
            if not self.gui_done:
                if policy_fn:
                    self.current_action = policy_fn(self.get_policy_state())
                    if self.current_action != 0: self.last_accel_action = self.current_action
                
                _, reward, done = self.step(self.current_action)
                self.gui_steps += 1
                
                canvas.delete("all")
                # Grid
                for i in range(int(self.grid_size) + 1):
                    canvas.create_line(i*cell_size, 0, i*cell_size, height, fill="#800000", dash=(2,4))
                    canvas.create_line(0, i*cell_size, width, i*cell_size, fill="#800000", dash=(2,4))
                
                # Goal & Holes
                gx_px, gy_px = self.goal_pos * cell_size
                canvas.create_rectangle(gx_px-15, gy_px+5, gx_px+15, gy_px+15, fill="#8B4513")
                canvas.create_polygon([gx_px-10, gy_px+5, gx_px, gy_px-20, gx_px+10, gy_px+5], fill="#FFA500")
                for hx, hy in self.holes:
                    hx_px, hy_px = hx*cell_size, hy*cell_size
                    # Draw hole with white outline and inner depth
                    canvas.create_oval(hx_px-20, hy_px-20, hx_px+20, hy_px+20, fill="black", outline="white", width=2)
                    canvas.create_oval(hx_px-14, hy_px-14, hx_px+14, hy_px+14, fill="#1a1a1a", outline="")

                # Rosie
                rx, ry, vx, vy, wm = self.state
                rx_px, ry_px = rx*cell_size, ry*cell_size
                angle = 0
                if self.last_accel_action == 1: angle = 90
                elif self.last_accel_action == 2: angle = -90
                elif self.last_accel_action == 3: angle = 0
                elif self.last_accel_action == 4: angle = 180
                else: angle = np.degrees(np.arctan2(vy, vx)) if abs(vx)>0.01 or abs(vy)>0.01 else 0

                # Ears
                for side in [-1, 1]:
                    ear_pts = []
                    # Center of head in GUI: rx_px + 18, ry_px
                    for d in range(0, 361, 30):
                        ex, ey = rotate_gui(rx_px + 18 + 4*np.cos(np.radians(d)), ry_px + side*8 + 6*np.sin(np.radians(d)), rx_px, ry_px, angle)
                        ear_pts.extend([ex, ey])
                    canvas.create_polygon(ear_pts, fill="gray", outline="black")

                # Body
                body_pts = []
                for d in range(0, 361, 30):
                    px, py = rotate_gui(rx_px + 18*np.cos(np.radians(d)), ry_px + 12*np.sin(np.radians(d)), rx_px, ry_px, angle)
                    body_pts.extend([px, py])
                canvas.create_polygon(body_pts, fill="gray", outline="black")

                # Legs
                for lx_off, ly_off in [(-10, -10), (10, -10), (-10, 10), (10, 10)]:
                    lx, ly = rotate_gui(rx_px + lx_off, ry_px + ly_off, rx_px, ry_px, angle)
                    canvas.create_rectangle(lx-3, ly-3, lx+3, ly+3, fill="gray", outline="black")
                
                # Head
                hx, hy = rotate_gui(rx_px + 18, ry_px, rx_px, ry_px, angle)
                hr = 9
                canvas.create_oval(hx-hr, hy-hr, hx+hr, hy+hr, fill="gray", outline="black")

                # Trunk & Water
                tx_s, ty_s = rotate_gui(rx_px+24, ry_px, rx_px, ry_px, angle)
                wdx, wdy = 0, 0
                if self.current_action == 1: wdy = 1
                elif self.current_action == 2: wdy = -1
                elif self.current_action == 3: wdx = 1
                elif self.current_action == 4: wdx = -1
                
                if self.current_action != 0 and wm > 0:
                    tx_e, ty_e = rotate_gui(rx_px+24+wdx*12, ry_px+wdy*12, rx_px, ry_px, angle)
                    for _ in range(5):
                        wx = tx_e + wdx*np.random.uniform(5, 25) + np.random.normal(0, 3)
                        wy = ty_e + wdy*np.random.uniform(5, 25) + np.random.normal(0, 3)
                        canvas.create_oval(wx-2, wy-2, wx+2, wy+2, fill="#ADD8E6", outline="")
                else: tx_e, ty_e = rotate_gui(rx_px+30, ry_px+6, rx_px, ry_px, angle)
                canvas.create_line(tx_s, ty_s, tx_e, ty_e, fill="gray", width=3)

                # Eye
                ex, ey = rotate_gui(rx_px + 21, ry_px - 3, rx_px, ry_px, angle)
                canvas.create_oval(ex-1, ey-1, ex+1, ey+1, fill="black")

                # Water meter
                canvas.create_rectangle(10, height-30, 110, height-10, outline="black")
                water_w = (wm / self.initial_water) * 100
                canvas.create_rectangle(10, height-30, 10 + water_w, height-10, fill="#ADD8E6", outline="")
                canvas.create_text(60, height-20, text=f"Water: {wm:.1f}kg")
                canvas.create_text(10, 10, anchor="nw", text=f"Steps: {self.gui_steps} | Pos: ({rx:.2f}, {ry:.2f})")

                if done:
                    self.gui_done = True
                    _, fin_rew = self._is_terminal(self.state[:2], wm)
                    msg = "Goal Reached!" if fin_rew > 0 else "Mission Failed!"
                    canvas.create_text(width/2, height/2, text=msg, font=("Arial", 24, "bold"), fill="#800000")
                    canvas.create_text(width/2, height/2 + 36, text="Press R or click Reset to play again",
                                       font=("Arial", 13), fill="#800000")
                else:
                    root.after(int(self.dt*1000), update)
        update()
        root.mainloop()

if __name__ == "__main__":
    env = RosieMomentumEnv()
    env.run_gui()
