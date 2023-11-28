#include "sdf_shapes.hpp"
#include "common.cu"

namespace GPUImpl::Sphere {
    __device__ __forceinline__ float
    get_signed_distance(const float point[3], SDFObject *sdf_object_gpu) {
        auto sdf_sphere_gpu = (SDFSphere *) sdf_object_gpu;
        float dx = point[0];
        float dy = point[1];
        float dz = point[2];

        return sqrt(dx * dx + dy * dy + dz * dz) - sdf_sphere_gpu->radius;
    }

    __device__ __forceinline__ void
    get_sdf_normal(const float point[3], float normal[3], SDFObject *sdf_object_gpu) {
        auto sdf_sphere_gpu = (SDFSphere *) sdf_object_gpu;
        float dx = point[0];
        float dy = point[1];
        float dz = point[2];
        float dist = sqrt(dx * dx + dy * dy + dz * dz);
        normal[0] = dx / dist;
        normal[1] = dy / dist;
        normal[2] = dz / dist;
    }

    // need to initialize function pointers in static memory
    __device__ get_signed_distance_t sdf_func_d = get_signed_distance;
    __device__ get_sdf_normal_t sdf_normal_func_d = get_sdf_normal;

    SDFObjectGPU create(SDFObject *sdf_object_cpu) {
        SDFObjectGPU out{*sdf_object_cpu};    //{sdf_func_d, sdf_normal_func_d, *sdf_object_cpu};
        cudaMemcpyFromSymbol(&out.get_signed_distance, sdf_func_d, sizeof(get_signed_distance_t));
        cudaMemcpyFromSymbol(&out.get_sdf_normal, sdf_normal_func_d, sizeof(get_sdf_normal_t));

        return std::move(out);
    }
}

SDFObjectGPU SDFSphere::createGPU() {
    SDFObjectGPU &&out = GPUImpl::Sphere::create(this);
    return std::move(out);
}
