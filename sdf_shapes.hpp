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

    std::shared_ptr<SDFObject> operator()() {
        auto obj_ptr = new SDFSphere;
        memcpy(obj_ptr->T, this->T.data(), 12 * sizeof(float));
        obj_ptr->radius = this->radius;
        return std::shared_ptr<SDFObject>((SDFObject *) obj_ptr);
    }

};

class SDFPolynomial : public SDFObject {
public:
    float *coefficients = nullptr;
    int num_coefficients;

    SDFObjectGPU createGPU() override;

    ~SDFPolynomial() {
        if (coefficients) {
            delete[] coefficients;
        }

    }

    float radius = 1;
};


class SDFPolynomialPy {
public:
    pybind11::array_t<float> T;
    pybind11::array_t<float> coefficients;

    SDFPolynomialPy(int num_coefficients) {
        float vec[12] = {1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0};
        T = pybind11::array_t<float>(12, vec);
        T.resize({{3, 4}});

        std::vector<float> vec_coeff;
        vec_coeff.assign(num_coefficients, 0);
        coefficients = pybind11::array_t<float>(vec_coeff.size(), vec_coeff.data());

    }

    std::shared_ptr<SDFObject> operator()() {
        auto obj_ptr = new SDFPolynomial;
        memcpy(obj_ptr->T, this->T.data(), 12 * sizeof(float));
        obj_ptr->coefficients = new float[this->coefficients.size()];
        memcpy(obj_ptr->coefficients, this->coefficients.data(), this->coefficients.size() * sizeof(float));
        obj_ptr->num_coefficients = this->coefficients.size();
        return std::shared_ptr<SDFObject>((SDFObject *) obj_ptr);
    }

};
