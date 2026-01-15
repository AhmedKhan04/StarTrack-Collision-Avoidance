import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 
import time 
MU_EARTH = 3.986004418e14

class Propogater:

    def __init__(self, mu, reverse = False):
        self.mu = MU_EARTH
        R_EARTH = 6378e3
        altitude = 500e3
        r_mag = R_EARTH + altitude
        v_circ = np.sqrt(MU_EARTH / r_mag)

        rA0 = np.array([r_mag, 0, 0])
        vA0 = np.array([0, v_circ, 10])
        if reverse:
            rA0 = np.array([r_mag, 0, 0])
            vA0 = np.array([0, -v_circ, -10])

        self.state0 = np.hstack((rA0, vA0))  # Initial state vector [rx, ry, rz, vx, vy, vz]
        #print(self.state0)
        self.solution_t, self.solution_y = self.solver(self.state0, (0, 9000))  # Propagate for 9000 seconds (150 minutes)

    def dynamics(self, t, state):
        #print(type(state))
        r = state[:3]
        v = state[3:]
        #print(r)
        #print(v)
        r_norm = np.linalg.norm(r)
        r_hat = r / r_norm
        accel = (-self.mu / r_norm**2 ) * r_hat
        state_dot = np.hstack((v, accel))
        return state_dot
    
    def RK4(self, t, state, dt):
        k1 = self.dynamics(t, state)
        k2 = self.dynamics(t + 0.5 * dt, state + 0.5 * dt * k1)
        k3 = self.dynamics(t + 0.5 * dt, state + 0.5 * dt * k2)
        k4 = self.dynamics(t + dt, state + dt * k3)
        state_next = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return state_next

    def solver(self, state, t_span): 

        solution =  sp.integrate.solve_ivp(fun = self.dynamics, t_span = t_span,   y0 = state, method='RK45', rtol=1e-9, atol=1e-12)  
        #print(solution)
   
        return solution.t, solution.y # returns time, [r0,r1,r2, v0,v1,v2]
    
    @staticmethod
    def Propogater_plotter(solution_y = None, figure = plt.figure(), ax = None):

        if solution_y is None:
            raise ValueError("solution_y must be provided for plotting.")
        

        #ax = figure.add_subplot(111, projection='3d')
        ax.plot(solution_y[0], solution_y[1], solution_y[2])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Orbit Propagation using Two-Body Problem')
        print("done! plotting")
        return 
        
    def save_solution_csv(self, filename=f"PROP_SAVED/two_body_solution_{int(time.time() * 1000)}"):
        df = pd.DataFrame({
            'time(s)': self.solution_t,
            'rx(m)': self.solution_y[0],
            'ry(m)': self.solution_y[1],
            'rz(m)': self.solution_y[2],
            'vx(m/s)': self.solution_y[3],
            'vy(m/s)': self.solution_y[4],
            'vz(m/s)': self.solution_y[5]
        })
        df.to_csv(f"{filename}_{np.random.randint(10000)}.csv", index=False)
        print(f"Solution saved to {filename}")

if __name__ == "__main__":
    figure2 = plt.figure()
    ax = figure2.add_subplot(111, projection='3d')

    print("----- Forward Propogation -----")
    propagator = Propogater(MU_EARTH)
    Propogater.Propogater_plotter(solution_y=propagator.solution_y, figure=figure2, ax=ax)
    propagator.save_solution_csv()

    print("----- Reversed Propogation -----")
    propagator_rev = Propogater(MU_EARTH, reverse=True)
    Propogater.Propogater_plotter(solution_y=propagator_rev.solution_y, figure=figure2, ax=ax)
    propagator_rev.save_solution_csv()
    
    plt.show()


    