import numpy as np
import sdf_experiments.sdf_experiments_py as sdf_experiments_py
import time

if __name__ == "__main__":
    # sdf_object = sdf_experiments_py.SDFSphere()
    # sdf_object.radius = 1.0
    # sdf_object.coefficients = np.random.rand(10)

    controller = sdf_experiments_py.ImguiController()

    tmp = np.meshgrid(np.linspace(-1., 1, 4), np.linspace(-1., 1, 4), np.linspace(-1., 1, 4), indexing='ij')
    centers = np.hstack([np.reshape(val, (-1, 1)) for val in tmp])
    sdf_object = sdf_experiments_py.SDFRadial(centers)
    sdf_object.coefficients = .29* (1 - np.random.rand(sdf_object.coefficients.size))
    sdf_object.coefficients[0] = -2.5

    sdf_object.T[0, 3] = 0
    sdf_object.T[1, 3] = 0.0
    sdf_object.T[2, 3] = -8.0

    while True:
        T = controller.get_camera_transform()
        sdf_object.T = T

        res_y = controller.get_height()
        res_x = controller.get_width()
        fy = 300
        fx = 300
        start = time.time_ns()
        img = sdf_experiments_py.render(fx, fy, res_x, res_y, sdf_object)
        # print((time.time_ns()-start)/1E6)
        img = img[:, :, :3]
        controller.set_img(img[::-1, :, :].copy())

        time.sleep(.05)
