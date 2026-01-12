import numpy as np
import pandas as pd 

import time 


class StreakMeasurement:
    def simulate_streak(r, v, exposure_time, fov_deg=8.0, resolution=1024, centroid_noise_px=0.2, length_noise_px=0.5):
        
        fov = np.radians(fov_deg)
        focal = (resolution / 2) / np.tan(fov / 2)



        r0 = r # start position
        r1 = r + (v * exposure_time) # end position

        
        r_0_hat = r0 / np.linalg.norm(r0)
        r_1_hat = r1 / np.linalg.norm(r1)

        # image plane coordinates
        def ToImage(u):
            return np.array([focal * u[0] / u[2], focal * u[1] / u[2]])

        p0 = ToImage(r_0_hat)
        p1 = ToImage(r_1_hat)

        
        centroid = 0.5 * (p0 + p1)
        delta = p1 - p0

        length = np.linalg.norm(delta)
        phi = np.arctan2(delta[1], delta[0])

        # add noise
        centroid += np.random.randn(2) * centroid_noise_px
        length += np.random.randn() * length_noise_px

        return centroid[0], centroid[1], length, phi
    

    # propagate this data over a batch of r,v inputs from orbital mechanics 
    
    def simulate_batch(input_csv, exposure_time, fov_deg=8.0, resolution=1024, centroid_noise_px=0.2, length_noise_px=0.5):
        centroid_x_measure = []
        centroid_y_measure = []
        length_measure = []
        phi_measure = []
        time_measure = []

        data = pd.read_csv(input_csv)
        r_batch = data[['r_x', 'r_y', 'r_z']].to_numpy()
        v_batch = data[['v_x', 'v_y', 'v_z']].to_numpy()
        time = data['time'].to_numpy() 

        for r, v, t in zip(r_batch, v_batch, time):
            try: 
                centroid_x, centroid_y, length, phi = StreakMeasurement.simulate_streak(r, v, exposure_time, fov_deg, resolution, centroid_noise_px, length_noise_px)
                centroid_x_measure.append(centroid_x)
                centroid_y_measure.append(centroid_y)
                length_measure.append(length)
                phi_measure.append(phi)
                time_measure.append(t)
            except Exception as e:
                print(f"Error simulating streak for r={r}, v={v} at time={t}: {e}")
                continue
        
        df = pd.DataFrame({
            'centroid_x': centroid_x_measure,
            'centroid_y': centroid_y_measure,
            'length': length_measure,
            'phi': phi_measure,
            'time': time_measure
        })

        df.to_csv(f'OUTPUT_SIMS/streak_measurements_{int(time.time() * 1000)}.csv', index=False)

        return df 
    

    