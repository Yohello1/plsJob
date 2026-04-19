#ifndef _VISCOSITY_K_HPP
#define _VISCOSITY_K_HPP

#include "struct.hpp"

namespace JD::Viscosity_k {
    float laplacian(float distance_i, float particle_size_i) 
    {
        float h = particle_size_i;
        float h2 = h * h;
        if (distance_i < 0 || distance_i >= h2) return 0.0f;

        float coeff = (1/( ::sycl::pow(h, 6))) * PARTICLE_VISCOSITY_K_COEFF;

        return coeff * (h - ::sycl::sqrt(distance_i)) ;
    }
};

#endif
