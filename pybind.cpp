#include "sdf_renderer.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <variant>

pybind11::array_t<uint8_t>
render(float fx, float fy, unsigned int res_x, unsigned int res_y, std::variant<SDFSpherePy> &sdf_object) {

    std::shared_ptr<SDFObject> object;
    if (SDFSpherePy *obj_py = std::get_if<SDFSpherePy>(&sdf_object)) {
        object = obj_py->operator()();
    } else {
        throw pybind11::type_error();
    }

    auto img_vec = internal::render(fx, fy, res_x, res_y, *object);
    pybind11::array_t<uint8_t> img(img_vec.size(), img_vec.data());
    img.resize({{res_x, res_y, 4}});

    return img;
}

PYBIND11_MODULE(sdf_experiments_py, m) {
    m.def("render", render);
    pybind11::class_<SDFSpherePy>(m, "SDFSphere", R"(
    SDFSphere contains parameters of SDF.
									     )")

            .def(pybind11::init([]() {
                     auto sdf_object = SDFSpherePy();
                     return sdf_object;
                 }),
                 R"(
                 Init.
           )").def("__str__", [](const SDFSpherePy &sdf_object) {
                std::stringstream ss;
                for (int r = 0; r < 3; r++) {
                    ss << "T:\n[";
                    for (int c = 0; c < 4; c++) {
                        ss << sdf_object.T.data()[r * 4 + c] << ", ";
                    }
                    ss << "]\n";
                }
                ss << "radius: " << sdf_object.radius;
                return ss.str();
            }).def_readwrite("T", &SDFSpherePy::T)
            .def_readwrite("radius", &SDFSpherePy::radius);
}