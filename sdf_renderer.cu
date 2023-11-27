#include "sdf_renderer.hpp"
#include <cstdio>

struct SDFObjectGPU {
    const SDFObject &sdf_object_cpu;
    SDFObject *sdf_object_gpu;

    explicit SDFObjectGPU(const SDFObject &sdf_object_in) : sdf_object_cpu{sdf_object_in} {
        cudaMalloc(&sdf_object_gpu, sizeof(SDFObject));
        cudaMemcpy(sdf_object_gpu, &sdf_object_cpu, sizeof(SDFObject), cudaMemcpyKind::cudaMemcpyHostToDevice);
    }

    ~SDFObjectGPU() {
        cudaFree(sdf_object_gpu);
    }
};

template<typename T>
struct BufferGPU {
    T *buffer;
    size_t size;

    explicit BufferGPU(size_t size_in) : size{size_in} {
        cudaMalloc(&buffer, size * sizeof(T));
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

namespace internal {
    __global__ void
    render_kernel(float fx, float fy, unsigned int res_x, unsigned int res_y, const SDFObject *const sdf_object_gpu,
                  unsigned char *img) {
        constexpr int stride = 4;
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int ind_x = idx % res_x;
        unsigned int ind_y = (idx / res_x) % res_y;


        if (idx < res_x * res_y) {
            img[stride * idx] = (255 * ind_x) / res_x;
            img[stride * idx + 1] = (256 * ind_y) / res_y;
            img[stride * idx + 2] = 0;
            img[stride * idx + 3] = 255;

        }


    }

    std::vector<unsigned char>
    render(float fx, float fy, unsigned int res_x, unsigned int res_y, const SDFObject &sdf_object_cpu) {
        SDFObjectGPU sdf_object_gpu{sdf_object_cpu};
        constexpr size_t block_size = 256;
        size_t num_threads = res_y * res_x;
        size_t grid_size = (num_threads + block_size - 1) / block_size;
        BufferGPU<unsigned char> img_buffer{res_x * res_y * 4};

        render_kernel<<<grid_size, block_size>>>(fx, fy, res_x, res_y, sdf_object_gpu.sdf_object_gpu,
                                                 img_buffer.buffer);

        auto img = img_buffer.toCPU();

        return img;
    }

}
