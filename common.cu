#include "sdf_shapes.hpp"

typedef float(*get_signed_distance_t)(const float point[3], SDFObject *sdf_object_gpu);

typedef void(*get_sdf_normal_t)(const float point[3], float normal[3], SDFObject *sdf_object_gpu);

class SDFObjectGPU {
public:
    get_signed_distance_t get_signed_distance;
    get_sdf_normal_t get_sdf_normal;
    SDFObject *sdf_object_gpu;

    SDFObjectGPU(const SDFObject &sdf_object_cpu){
        cudaMalloc(&sdf_object_gpu, sizeof(SDFObject));
        cudaMemcpy(sdf_object_gpu, &sdf_object_cpu, sizeof(SDFObject), cudaMemcpyKind::cudaMemcpyHostToDevice);
    };

    SDFObjectGPU(SDFObjectGPU&& other){
        get_signed_distance = other.get_signed_distance;
        get_sdf_normal = other.get_sdf_normal;
        sdf_object_gpu = other.sdf_object_gpu;
        other.get_signed_distance = nullptr;
        other.get_sdf_normal = nullptr;
        other.sdf_object_gpu = nullptr;
    }

    ~SDFObjectGPU() {
        if (sdf_object_gpu){
            cudaFree(sdf_object_gpu);
        }
    }
};