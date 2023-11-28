import sdf_experiments.sdf_experiments_py as sdf_experiments_py
from PIL import Image

if __name__ == "__main__":
    sdf_object = sdf_experiments_py.SDFObject()
    sdf_object.radius = 1
    sdf_object.x = 0
    sdf_object.y = 0
    sdf_object.z = -3.0

    res_y = 300
    res_x = 300
    fx = 150
    fy = 150
    img = sdf_experiments_py.render(fx, fy, res_x, res_y, sdf_object)
    image = Image.fromarray(img[:, :, :4])
    image.show()
    pass
