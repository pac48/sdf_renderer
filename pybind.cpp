#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sstream>
#include "sdf_renderer.hpp"

pybind11::array_t<uint8_t> render(float fx, float fy, unsigned int res_x, unsigned int res_y, SDFObject &sdf_object) {
    auto img_vec = internal::render(fx, fy, res_x, res_y, sdf_object);
    pybind11::array_t<uint8_t> img(img_vec.size(), img_vec.data());
    img.resize({{res_x, res_y, 4}});

    return img;
}

PYBIND11_MODULE(sdf_experiments_py, m) {
    m.def("render", render);
    pybind11::class_<SDFObject>(m, "SDFObject", R"(
    SDFObject contains parameters of SDF.
									     )")

            .def(pybind11::init([]() {
                     auto sdf_object = SDFObject();
                     return sdf_object;
                 }),
                 R"(
                 Init.
           )").def("__str__", [](const SDFObject &sdf_object) {
                std::stringstream ss;
                ss << sdf_object.x << " " << sdf_object.y << " " << sdf_object.z << " " << sdf_object.radius << " ";
                return ss.str();
            }).def_readwrite("x", &SDFObject::x)
            .def_readwrite("y", &SDFObject::y)
            .def_readwrite("z", &SDFObject::z)
            .def_readwrite("radius", &SDFObject::radius);
}