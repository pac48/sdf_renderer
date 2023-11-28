#include "vector"

// use sphere for now
struct SDFObject {
    float x = 0.0;
    float y = 0.0;
    float z = 0.0;
    float radius = 1;
    float T[12] = {1.0, 0.0, 0.0, 0.0,
                   0.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 1.0, 0.0};
};

namespace internal {
    std::vector<unsigned char>
    render(float fx, float fy, unsigned int res_x, unsigned int res_y, SDFObject &sdf_object);
}
