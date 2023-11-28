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
        SDFObjectGPU out;    //{sdf_func_d, sdf_normal_func_d, *sdf_object_cpu};

        SDFObjectGPU::GPUData gpu_data_local;
        cudaMalloc(&gpu_data_local.sdf_object_gpu, sizeof(SDFSphere));
        cudaMemcpy(gpu_data_local.sdf_object_gpu, sdf_object_cpu, sizeof(SDFSphere),
                   cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMalloc(&out.gpu_data, sizeof(SDFObjectGPU::GPUData));
        cudaMemcpy(out.gpu_data, &gpu_data_local, sizeof(SDFObjectGPU::GPUData), cudaMemcpyKind::cudaMemcpyHostToDevice);

        cudaMemcpyFromSymbol(&out.gpu_data->get_signed_distance, sdf_func_d, sizeof(get_signed_distance_t));
        cudaMemcpyFromSymbol(&out.gpu_data->get_sdf_normal, sdf_normal_func_d, sizeof(get_sdf_normal_t));

        return std::move(out);
    }
}

SDFObjectGPU SDFSphere::createGPU() {
    SDFObjectGPU &&out = GPUImpl::Sphere::create(this);
    return std::move(out);
}

std::shared_ptr<SDFObject> SDFSpherePy::operator()() {
    auto obj_ptr = new SDFSphere;
    memcpy(obj_ptr->T, this->T.data(), 12 * sizeof(float));
    obj_ptr->radius = this->radius;
    return std::shared_ptr<SDFObject>((SDFObject *) obj_ptr);
}
