#include "common.cu"
#include "sdf_renderer.hpp"
#include "sdf_shapes.hpp"
#include <cstdio>

template<typename T>
struct BufferGPU {
    T *buffer;
    size_t size;

    explicit BufferGPU(size_t size_in) : size{size_in} {
        cudaMalloc(&buffer, size * sizeof(T));
    }

    BufferGPU(const BufferGPU &other) {
        size = other.size;
        cudaMalloc(&buffer, size * sizeof(T));
    }

    BufferGPU &operator=(const BufferGPU &other) {
        cudaFree(buffer);
        size = other.size;
        cudaMalloc(&buffer, size * sizeof(T));
        return *this;
    }

    ~BufferGPU() {
        cudaFree(buffer);
    }

    std::vector<T> toCPU() {
        std::vector<T> out(size);
        cudaMemcpy(out.data(), buffer, size * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost);

        return out;
    }

};

std::shared_ptr<BufferGPU<unsigned char>> gpu_buffer = nullptr;


namespace internal {

    __device__ __forceinline__ void transform_mul(const float *const T1, const float *const T2, float *T_out) {
        for (int i = 0; i < 12; i++) {
            T_out[i] = 0.0;
        }

#pragma unroll
        for (int k = 0; k < 4; k++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
#pragma unroll
                for (int i = 0; i < 3; i++) {
                    T_out[i * 4 + j] += T1[i * 4 + k] * T2[k * 3 + j];
                }
            }
        }
    }

    __device__ __forceinline__ void transform_vec(const float *const T, const float *const vec_in, float *vec_out) {
        for (int i = 0; i < 3; i++) {
            vec_out[i] = 0.0;
        }

#pragma unroll
        for (int i = 0; i < 3; i++) {
#pragma unroll
            for (int j = 0; j < 3; j++) {
                vec_out[i] += T[i * 4 + j] * vec_in[j];
            }
            vec_out[i] += T[i * 4 + 3];
        }
    }

    __device__ __forceinline__ void rotate_vec(const float *const T, const float *const vec_in, float *vec_out) {
        for (int i = 0; i < 3; i++) {
            vec_out[i] = 0.0;
        }

#pragma unroll
        for (int i = 0; i < 3; i++) {
#pragma unroll
            for (int j = 0; j < 3; j++) {
                vec_out[i] += T[i * 4 + j] * vec_in[j];
            }
        }
    }

    __global__ void
    render_kernel(float fx, float fy, int res_x, int res_y,
                  const SDFObjectGPU::GPUData *const sdf_object_gpu,
                  unsigned char *img) {
        constexpr int stride = 4;
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int ind_x = idx % res_x;
        int ind_y = (idx / res_x) % res_y;

        if (idx < res_x * res_y) {

            float near_dist = 0.01;
            float far_dist = 100.0;
            float thresh = 0.001;
            float dist = 1.0;
            float scale = 0.1;
            float brightness = 200;

            float delta_orig[3] = {near_dist * (ind_x - res_x / 2) / fx, near_dist * (ind_y - res_y / 2) / fy,
                                   near_dist};
            float light_dir[3] = {sdf_object_gpu->sdf_object_gpu->T[2], sdf_object_gpu->sdf_object_gpu->T[6],
                                  sdf_object_gpu->sdf_object_gpu->T[10]};
            float point[3];
            float dir[3];

            transform_vec(sdf_object_gpu->sdf_object_gpu->T, delta_orig, point);
            rotate_vec(sdf_object_gpu->sdf_object_gpu->T, delta_orig, dir);

            float dir_len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
            float dir_x = dir[0] / dir_len;
            float dir_y = dir[1] / dir_len;
            float dir_z = dir[2] / dir_len;

            while (dist > thresh && dist < far_dist) {
                dist = scale * sdf_object_gpu->get_signed_distance(point, sdf_object_gpu->sdf_object_gpu);
                if (dist > 0) {
                    point[0] += dist * dir_x;
                    point[1] += dist * dir_y;
                    point[2] += dist * dir_z;

                }
            }

            img[stride * idx + 3] = 255;
            if (dist <= thresh) {
                float normal[3];
                sdf_object_gpu->get_sdf_normal(point, normal, sdf_object_gpu->sdf_object_gpu);
                float res = normal[0] * light_dir[0] + normal[1] * light_dir[1] + normal[2] * light_dir[2];
                brightness *= max(-res, 0.03f);
                img[stride * idx] = brightness;
                img[stride * idx + 1] = brightness;
                img[stride * idx + 2] = brightness;
//                printf("hit! point: [%f, %f, %f]\n", point[0], point[1], point[2]);
            } else {
                img[stride * idx] = 0;
                img[stride * idx + 1] = 0;
                img[stride * idx + 2] = 0;
//                printf("miss! point: [%f, %f, %f]\n", point[0], point[1], point[2]);
            }
        }

    }

    std::vector<unsigned char>
    render(float fx, float fy, unsigned int res_x, unsigned int res_y, SDFObject &sdf_object_cpu) {
        auto sdf_object_gpu = sdf_object_cpu.createGPU();
        constexpr size_t block_size = 256;
        size_t num_threads = res_y * res_x;
        size_t grid_size = (num_threads + block_size - 1) / block_size;

        if (gpu_buffer== nullptr || res_x * res_y * 4 > gpu_buffer->size ) {
            gpu_buffer = std::make_shared<BufferGPU<unsigned char>>(res_x * res_y * 4);
        }

        render_kernel<<<grid_size, block_size>>>(fx, fy, res_x, res_y, sdf_object_gpu->gpu_data,
                                                 gpu_buffer->buffer);

        auto img = gpu_buffer->toCPU();

        return img;
    }

}
