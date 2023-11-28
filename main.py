import sdf_experiments.sdf_experiments_py as sdf_experiments_py
from PIL import Image

if __name__ == "__main__":
    sdf_object = sdf_experiments_py.SDFSphere()
    sdf_object.radius = 1.25
    sdf_object.x = 0.0
    sdf_object.y = 0.0
    sdf_object.z = -3.0

    res_y = 3000
    res_x = 3000
    fx = 1500
    fy = 1500
    img = sdf_experiments_py.render(fx, fy, res_x, res_y, sdf_object)
    image = Image.fromarray(img[:, :, :4])
    image.show()
    pass
    input()
