#ifndef JD_VISCOSITY_K_HPP
#define JD_VISCOSITY_K_HPP

#include <sycl/sycl.hpp>

#include "settings.hpp"
#include "struct.hpp"

namespace JD::Viscosity_k
{
    inline float laplacian(float distance_squared, float particle_size)
    {
        const float h2 = particle_size * particle_size;
        if (distance_squared <= 0.0f || distance_squared >= h2) {
            return 0.0f;
        }
        const float coefficient = PARTICLE_VISCOSITY_K_COEFF / (h2 * h2 * h2);
        return coefficient * (particle_size - ::sycl::sqrt(distance_squared));
    }
}

#endif
