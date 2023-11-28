#pragma once

class SDFObjectGPU;

class SDFObject {
public:
    float x = 0.0;
    float z = 0.0;
    float y = 0.0;
    float T[12] = {1.0, 0.0, 0.0, 0.0,
                   0.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 1.0, 0.0};

    virtual SDFObjectGPU createGPU() = 0;

};

class SDFSphere : public SDFObject {
public:
    SDFObjectGPU createGPU() override;

    float radius = 1;
};
