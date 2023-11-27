import sdf_experiments.sdf_experiments_py as sdf_experiments_py
from PIL import Image
import numpy as np

if __name__ == "__main__":
    sdf_object = sdf_experiments_py.SDFObject()
    sdf_object.radius = .5
    res_y = 300
    res_x = 300
    fx = 500
    fy = 500
    img = sdf_experiments_py.render(fx, fy, res_x, res_y, sdf_object)
    # img = np.reshape(img, (4, res_x, res_y))
    # img = img.transpose(1, 2, 0)

    img = np.reshape(img, (res_x, res_y, 4))

    image = Image.fromarray(img[:, :, :4])
    image.show()
    pass
