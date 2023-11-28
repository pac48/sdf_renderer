#pragma once

#include <pybind11/numpy.h>

class SDFObjectGPU;

class SDFObject {
public:
    float T[12] = {
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0
    };

    virtual SDFObjectGPU createGPU() = 0;


};


class SDFSphere : public SDFObject {
public:

    SDFObjectGPU createGPU() override;

    float radius = 1;
};


class SDFSpherePy {
public:
    pybind11::array_t<float> T;
    float radius = 1;

    SDFSpherePy() {
        float vec[12] = {1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0};
        T = pybind11::array_t<float>(12, vec);
        T.resize({{3, 4}});

    }

    std::shared_ptr<SDFObject> operator()();

};

