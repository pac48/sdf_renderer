#include "vector"

// use sphere for now
struct SDFObject {
    float x = 0;
    float y = 0;
    float z = 0;
    float radius = 1;
};

namespace internal {
    std::vector<unsigned char> render(float fx, float fy, unsigned int res_x, unsigned int res_y, const SDFObject &sdf_object);
}
