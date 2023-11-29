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
        cudaMemcpy(out.gpu_data, &gpu_data_local, sizeof(SDFObjectGPU::GPUData),
                   cudaMemcpyKind::cudaMemcpyHostToDevice);

        cudaMemcpyFromSymbol(&out.gpu_data->get_signed_distance, sdf_func_d, sizeof(get_signed_distance_t));
        cudaMemcpyFromSymbol(&out.gpu_data->get_sdf_normal, sdf_normal_func_d, sizeof(get_sdf_normal_t));

        return std::move(out);
    }
}

SDFObjectGPU SDFSphere::createGPU() {
    SDFObjectGPU &&out = GPUImpl::Sphere::create(this);
    return std::move(out);
}


namespace GPUImpl::Polynomial {
    __device__ __forceinline__ float
    get_signed_distance(const float point[3], SDFObject *sdf_object_gpu) {
        //TODO
        auto sdf_polynomial_gpu = (SDFPolynomial *) sdf_object_gpu;
        float dist = -1;
        return dist;
    }

    __device__ __forceinline__ void
    get_sdf_normal(const float point[3], float normal[3], SDFObject *sdf_object_gpu) {
        //TODO
        auto sdf_polynomial_gpu = (SDFPolynomial *) sdf_object_gpu;
        normal[0] = 0.0;
        normal[1] = 0;
        normal[2] = -1.0;
    }

    // need to initialize function pointers in static memory
    __device__ get_signed_distance_t sdf_func_d = get_signed_distance;
    __device__ get_sdf_normal_t sdf_normal_func_d = get_sdf_normal;

    SDFObjectGPU create(SDFPolynomial *sdf_object_cpu) {
        SDFObjectGPU out;
        //TODO params need to be copied to GPU

        SDFObjectGPU::GPUData gpu_data_local;
        cudaMalloc(&gpu_data_local.sdf_object_gpu, sizeof(SDFPolynomial));
        cudaMemcpy(gpu_data_local.sdf_object_gpu, sdf_object_cpu, sizeof(SDFPolynomial),
                   cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMalloc(&out.gpu_data, sizeof(SDFObjectGPU::GPUData));
        cudaMemcpy(out.gpu_data, &gpu_data_local, sizeof(SDFObjectGPU::GPUData),
                   cudaMemcpyKind::cudaMemcpyHostToDevice);

        cudaMemcpyFromSymbol(&out.gpu_data->get_signed_distance, sdf_func_d, sizeof(get_signed_distance_t));
        cudaMemcpyFromSymbol(&out.gpu_data->get_sdf_normal, sdf_normal_func_d, sizeof(get_sdf_normal_t));

        return std::move(out);
    }
}

SDFObjectGPU SDFPolynomial::createGPU() {
    SDFObjectGPU &&out = GPUImpl::Polynomial::create(this);
    return std::move(out);
}


namespace GPUImpl::Radial {
    __device__ __forceinline__ float
    get_signed_distance(const float point[3], SDFObject *sdf_object_gpu) {
        float dx = point[0];
        float dy = point[1];
        float dz = point[2];
        float dist = sqrt(dx * dx + dy * dy + dz * dz);

        auto sdf_radial_gpu = (SDFRadial *) sdf_object_gpu;
        dist += sdf_radial_gpu->coefficients[0]; //bias

        for (int i = 1; i < sdf_radial_gpu->num_coefficients; i++) {
            dx = sdf_radial_gpu->centers[3 * i] - point[0];
            dy = sdf_radial_gpu->centers[3 * i + 1] - point[1];
            dz = sdf_radial_gpu->centers[3 * i + 2] - point[2];
            dist += sdf_radial_gpu->coefficients[i] * expf((-1.0f / (2.0f * .1f)) * (dx * dx + dy * dy + dz * dz));
        }

        return dist;
    }

    __device__ __forceinline__ void
    get_sdf_normal(const float point[3], float normal[3], SDFObject *sdf_object_gpu) {
        //TODO
        float dx = point[0];
        float dy = point[1];
        float dz = point[2];
        float dist = sqrt(dx * dx + dy * dy + dz * dz);
        normal[0] = 0*dx / dist;
        normal[1] = 0*dy / dist;
        normal[2] = 0*dz / dist;

        auto sdf_radial_gpu = (SDFRadial *) sdf_object_gpu;


        for (int i = 1; i < sdf_radial_gpu->num_coefficients; i++) {
            dx = sdf_radial_gpu->centers[3 * i] - point[0];
            dy = sdf_radial_gpu->centers[3 * i + 1] - point[1];
            dz = sdf_radial_gpu->centers[3 * i + 2] - point[2];
            float tmp = sdf_radial_gpu->coefficients[i] * expf((-1.0f / (2.0f * .1f)) * (dx * dx + dy * dy + dz * dz));
            normal[0] += (-1.0f / .1f) * dx * tmp;
            normal[1] += (-1.0f / .1f) * dy * tmp;
            normal[2] += (-1.0f / .1f) * dz * tmp; // negative?
        }
        float scale = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        normal[0] /= scale;
        normal[1] /= scale;
        normal[2] /= scale;
    }

    // need to initialize function pointers in static memory
    __device__ get_signed_distance_t sdf_func_d = get_signed_distance;
    __device__ get_sdf_normal_t sdf_normal_func_d = get_sdf_normal;

    SDFObjectGPU create(SDFRadial *sdf_object_cpu) {
        SDFObjectGPU out;    //{sdf_func_d, sdf_normal_func_d, *sdf_object_cpu};

        SDFRadial sdf_object_cpu_copy = *sdf_object_cpu;
        cudaMalloc(&sdf_object_cpu_copy.centers, sdf_object_cpu_copy.num_coefficients * sizeof(float) * 3);
        cudaMemcpy(sdf_object_cpu_copy.centers, sdf_object_cpu->centers,
                   sdf_object_cpu_copy.num_coefficients * sizeof(float) * 3,
                   cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMalloc(&sdf_object_cpu_copy.coefficients, sdf_object_cpu_copy.num_coefficients * sizeof(float));
        cudaMemcpy(sdf_object_cpu_copy.coefficients, sdf_object_cpu->coefficients,
                   sdf_object_cpu_copy.num_coefficients * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);


        SDFObjectGPU::GPUData gpu_data_local;
        cudaMalloc(&gpu_data_local.sdf_object_gpu, sizeof(SDFRadial));
        cudaMemcpy(gpu_data_local.sdf_object_gpu, &sdf_object_cpu_copy, sizeof(SDFRadial),
                   cudaMemcpyKind::cudaMemcpyHostToDevice);

        cudaMalloc(&out.gpu_data, sizeof(SDFObjectGPU::GPUData));
        cudaMemcpy(out.gpu_data, &gpu_data_local, sizeof(SDFObjectGPU::GPUData),
                   cudaMemcpyKind::cudaMemcpyHostToDevice);

        cudaMemcpyFromSymbol(&out.gpu_data->get_signed_distance, sdf_func_d, sizeof(get_signed_distance_t));
        cudaMemcpyFromSymbol(&out.gpu_data->get_sdf_normal, sdf_normal_func_d, sizeof(get_sdf_normal_t));

        sdf_object_cpu_copy.centers = nullptr;
        sdf_object_cpu_copy.coefficients = nullptr;

        return std::move(out);
    }
}

SDFObjectGPU SDFRadial::createGPU() {
    SDFObjectGPU &&out = GPUImpl::Radial::create(this);
    return std::move(out);
}