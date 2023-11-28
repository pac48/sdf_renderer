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


    SDFObjectGPU() {
//        GPUData gpu_data_local;
//        cudaMalloc(&gpu_data_local.sdf_object_gpu, sizeof(SDFObject));
//        cudaMemcpy(gpu_data_local.sdf_object_gpu, sdf_object_cpu, sizeof(SDFObject),
//                   cudaMemcpyKind::cudaMemcpyHostToDevice);
//        cudaMalloc(&gpu_data, sizeof(GPUData));
//        cudaMemcpy(gpu_data, &gpu_data_local, sizeof(GPUData), cudaMemcpyKind::cudaMemcpyHostToDevice);
    };

    SDFObjectGPU(SDFObjectGPU &&other) {
//        gpu_data->get_signed_distance = other.gpu_data->get_signed_distance;
//        gpu_data->get_sdf_normal = other.gpu_data->get_sdf_normal;
//        gpu_data->sdf_object_gpu = other.gpu_data->sdf_object_gpu;
//        other.gpu_data->get_signed_distance = nullptr;
//        other.gpu_data->get_sdf_normal = nullptr;
//        other.gpu_data->sdf_object_gpu = nullptr;
        gpu_data = other.gpu_data;
        other.gpu_data = nullptr;
    }

    ~SDFObjectGPU() {
        if (gpu_data) {
            cudaFree(gpu_data);
        }
    }
};