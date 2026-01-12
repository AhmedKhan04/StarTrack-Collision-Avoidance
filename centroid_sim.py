import numpy as np

def simulate_streak(r, v, exposure_time,
                    fov_deg=8.0,
                    resolution=1024,
                    centroid_noise_px=0.2,
                    length_noise_px=0.5):
    """
    Simulates a star-tracker streak measurement.

    Inputs
    ------
    r : np.array(3,)  relative position [m]
    v : np.array(3,)  relative velocity [m/s]
    exposure_time : float [s]

    Returns
    -------
    z : np.array(4,)
        [u, v, L, phi]
        centroid px, centroid py, length px, angle rad
    """

    # --- Camera parameters ---
    fov = np.deg2rad(fov_deg)
    f = (resolution / 2) / np.tan(fov / 2)

    # --- Start and end positions ---
    r0 = r
    r1 = r + v * exposure_time

    # Normalize directions
    u0 = r0 / np.linalg.norm(r0)
    u1 = r1 / np.linalg.norm(r1)

    # Projection to image plane
    def project(u):
        return np.array([
            f * u[0] / u[2],
            f * u[1] / u[2]
        ])

    p0 = project(u0)
    p1 = project(u1)

    # --- Streak properties ---
    centroid = 0.5 * (p0 + p1)
    delta = p1 - p0

    length = np.linalg.norm(delta)
    phi = np.arctan2(delta[1], delta[0])

    # --- Add noise ---
    centroid += np.random.randn(2) * centroid_noise_px
    length += np.random.randn() * length_noise_px

    return np.array([centroid[0], centroid[1], length, phi])
