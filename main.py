import numpy as np
import sdf_experiments.sdf_experiments_py as sdf_experiments_py
from PIL import Image

if __name__ == "__main__":
    # sdf_object = sdf_experiments_py.SDFSphere()
    # sdf_object.radius = 1.0
    # sdf_object.coefficients = np.random.rand(10)

    tmp = np.meshgrid(np.linspace(-1., 1, 10), np.linspace(-1., 1, 10), np.linspace(-1., 1, 10), indexing='ij')
    centers = np.hstack([np.reshape(val, (-1, 1)) for val in tmp])
    sdf_object = sdf_experiments_py.SDFRadial(centers)
    sdf_object.coefficients = 10 * (.5 - np.random.rand(sdf_object.coefficients.size))
    sdf_object.coefficients[0] = 0

    sdf_object.T[0, 3] = 0
    sdf_object.T[1, 3] = 0.0
    sdf_object.T[2, 3] = -4.0

    res_y = 3000
    res_x = 3000
    fx = 1500
    fy = 1500
    img = sdf_experiments_py.render(fx, fy, res_x, res_y, sdf_object)
    image = Image.fromarray(img[:, :, :4])
    image.show()
    pass
    input()
