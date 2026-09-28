#ifndef JD_SPIKY_K_HPP
#define JD_SPIKY_K_HPP

#include <sycl/sycl.hpp>

#include "settings.hpp"
#include "struct.hpp"

namespace JD::Spiky_k
{
    inline void gradient(float dx, float dy, float distance_squared, float particle_size, force& result)
    {
        const float h2 = particle_size * particle_size;
        if (distance_squared <= 0.0f || distance_squared >= h2) {
            result = {0.0f, 0.0f};
            return;
        }
        const float distance = ::sycl::sqrt(distance_squared);
        const float difference = particle_size - distance;
        const float h3 = particle_size * particle_size * particle_size;
        const float h6 = h3 * h3;
        const float coefficient = -45.0f / (PI * h6);
        const float scalar = coefficient * difference * difference / (distance + 1.0e-6f);
        result.x = scalar * dx;
        result.y = scalar * dy;
    }
}

#endif
