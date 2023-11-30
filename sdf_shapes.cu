#include "sdf_shapes.hpp"
#include "common.cu"

template<typename T>
void allocate_gpu_data(const T *sdf_object_cpu, SDFObjectGPU::GPUData *&gpu_data) {
    SDFObjectGPU::GPUData gpu_data_local;
    cudaMalloc(&gpu_data_local.sdf_object_gpu, sizeof(T));
    cudaMemcpy(gpu_data_local.sdf_object_gpu, sdf_object_cpu, sizeof(T),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMalloc(&gpu_data, sizeof(SDFObjectGPU::GPUData));
    cudaMemcpy(gpu_data, &gpu_data_local, sizeof(SDFObjectGPU::GPUData),
               cudaMemcpyKind::cudaMemcpyHostToDevice);
}

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

    class SDFSphereGPU : public SDFObjectGPU {
    public:
        SDFSphereGPU(SDFSphere *sdf_object_cpu) {
            allocate_gpu_data(sdf_object_cpu, gpu_data);

            cudaMemcpyFromSymbol(&gpu_data->get_signed_distance, sdf_func_d, sizeof(get_signed_distance_t));
            cudaMemcpyFromSymbol(&gpu_data->get_sdf_normal, sdf_normal_func_d, sizeof(get_sdf_normal_t));

        }


        ~SDFSphereGPU() {
            //TODO
        }
    };
}

std::shared_ptr<SDFObjectGPU> SDFSphere::createGPU() {
    auto out = std::shared_ptr<SDFObjectGPU>(new GPUImpl::Sphere::SDFSphereGPU(this));
    return out;
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

    class SDFPolynomialGPU : public SDFObjectGPU {
    public:
        SDFPolynomialGPU(SDFPolynomial *sdf_object_cpu) {

            allocate_gpu_data(sdf_object_cpu, gpu_data);

            cudaMemcpyFromSymbol(&gpu_data->get_signed_distance, sdf_func_d, sizeof(get_signed_distance_t));
            cudaMemcpyFromSymbol(&gpu_data->get_sdf_normal, sdf_normal_func_d, sizeof(get_sdf_normal_t));
        }

        ~SDFPolynomialGPU() {
            //TODO
        }
    };
}

std::shared_ptr<SDFObjectGPU> SDFPolynomial::createGPU() {
    auto out = std::shared_ptr<SDFObjectGPU>(new GPUImpl::Polynomial::SDFPolynomialGPU(this));
    return out;
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
        normal[0] = 0 * dx / dist;
        normal[1] = 0 * dy / dist;
        normal[2] = 0 * dz / dist;

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

    class SDFRadialGPU : public SDFObjectGPU {
    public:
        SDFRadialGPU();

        SDFRadialGPU(SDFRadial *sdf_object_cpu) {

//            SDFRadial sdf_object_cpu_copy = *sdf_object_cpu;
            float *old_center = sdf_object_cpu->centers;
            float *old_coefficients = sdf_object_cpu->coefficients;

            cudaMalloc(&sdf_object_cpu->centers, sdf_object_cpu->num_coefficients * sizeof(float) * 3);
            cudaMemcpy(sdf_object_cpu->centers, old_center,
                       sdf_object_cpu->num_coefficients * sizeof(float) * 3,
                       cudaMemcpyKind::cudaMemcpyHostToDevice);
            cudaMalloc(&sdf_object_cpu->coefficients, sdf_object_cpu->num_coefficients * sizeof(float));
            cudaMemcpy(sdf_object_cpu->coefficients, old_coefficients,
                       sdf_object_cpu->num_coefficients * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

            allocate_gpu_data(sdf_object_cpu, gpu_data);

            cudaMemcpyFromSymbol(&gpu_data->get_signed_distance, sdf_func_d, sizeof(get_signed_distance_t));
            cudaMemcpyFromSymbol(&gpu_data->get_sdf_normal, sdf_normal_func_d, sizeof(get_sdf_normal_t));

            sdf_object_cpu->centers = old_center;
            sdf_object_cpu->coefficients = old_coefficients;

        }

        ~SDFRadialGPU() {
            SDFObjectGPU::GPUData gpu_data_local;
            cudaMemcpy(&gpu_data_local, gpu_data, sizeof(SDFObjectGPU::GPUData),
                       cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaFree(gpu_data);

            SDFRadial sdf_object_gpu;
            cudaMemcpy(&sdf_object_gpu, gpu_data_local.sdf_object_gpu, sizeof(SDFRadial),
                       cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaFree(sdf_object_gpu.centers);
            cudaFree(sdf_object_gpu.coefficients);
            cudaFree(gpu_data_local.sdf_object_gpu);
            sdf_object_gpu.centers = nullptr;
            sdf_object_gpu.coefficients = nullptr;
        }
    };
}

std::shared_ptr<SDFObjectGPU> SDFRadial::createGPU() {
    auto out = std::shared_ptr<SDFObjectGPU>(new GPUImpl::Radial::SDFRadialGPU(this));
    return out;
}