#include "sdf_shapes.hpp"

typedef float(*get_signed_distance_t)(const float point[3], SDFObject *sdf_object_gpu);

typedef void(*get_sdf_normal_t)(const float point[3], float normal[3], SDFObject *sdf_object_gpu);

class SDFObjectGPU {
public:
    struct GPUData {
        get_signed_distance_t get_signed_distance = nullptr;
        get_sdf_normal_t get_sdf_normal = nullptr;
        SDFObject *sdf_object_gpu = nullptr;
    };
    GPUData *gpu_data = nullptr;


    SDFObjectGPU() {};

    SDFObjectGPU(SDFObjectGPU &&other) {
        gpu_data = other.gpu_data;
        other.gpu_data = nullptr;
    }

    ~SDFObjectGPU() {}
};