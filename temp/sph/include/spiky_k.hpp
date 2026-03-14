#ifndef _SPIKY_K_HPP
#define _SPIKY_K_HPP

#include "struct.hpp" 

namespace JD::Spiky_k {
    inline void gradient(float dx, float dy, float dist_sq, float h, force& out_force) 
    {
        float h2 = h * h;
        if (dist_sq <= 0.0f || dist_sq >= h2) {
            out_force = {0.0f, 0.0f};
            return;
        }

        float dist = std::sqrt(dist_sq);
        float inv_dist = 1.0f / dist;

        float diff = h - dist;
        float scalar = (PARTICLE_SPIKY_K * diff * diff) * inv_dist;

        out_force.x = scalar * dx;
        out_force.y = scalar * dy;
    }
};

#endif
