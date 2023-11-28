#include "sdf_shapes.hpp"
#include "vector"

namespace internal {
    std::vector<unsigned char>
    render(float fx, float fy, unsigned int res_x, unsigned int res_y, SDFObject &sdf_object);
}
