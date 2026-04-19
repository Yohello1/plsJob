#ifndef _SPIKY_K_HPP
#define _SPIKY_K_HPP

#include "struct.hpp" 

namespace JD::Spiky_k {
    inline void gradient(float dx, float dy, float distance_i, float particle_size_i, force& out_force)
    {
        float h = particle_size_i;
        float h2 = h * h;
        if (distance_i <= 0 || distance_i >= h2) {
            out_force.x = 0.0f; out_force.y = 0.0f;
            return;
        }

        float r = sycl::sqrt(distance_i);
        float diff = h - r;

        const float pi = 3.14159265358979323846f;
        float h3 = h * h * h;
        float h6 = h3 * h3;
        float coeff = -45.0f / (pi * h6);

        float scalar = (coeff * diff * diff) / (r + 1e-6f);
        out_force.x = scalar * dx;
        out_force.y = scalar * dy;
    }
};

#endif
