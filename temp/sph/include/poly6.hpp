#ifndef _POLY6_HPP
#define _POLY6_HPP

#include "struct.hpp"



namespace JD::Poly6_k
{
    inline float smoothing(float distance_i, float h2) // Pass h2 in directly if you can!
    {
        float diff = h2 - distance_i;
        
        if (distance_i < 0.0f || distance_i >= h2) return 0.0f;

        return PARTICLE_POLY6_K_SMOOTHING * (diff * diff * diff);

    }
    // we modify the supplied struct for the sake of speed
    
    inline void gradient(float dx, float dy, float distance_i, float particle_size_i, force& out_force)
    {
        float h = particle_size_i;
        float h2 = h * h;
        if (distance_i <= 0 || distance_i >= h2)
        {
            out_force.x = 0.0f; out_force.y = 0.0f;
            return;
        }

        float diff = h2 - distance_i;
        float coeff = 4.0f / (std::numbers::pi_v<float> * std::pow(h, 8));

        float scalar = coeff * -6.0f * (diff * diff);

        // return~ish
        out_force.x = scalar * dx;
        out_force.y = scalar * dy;
    }
    inline float laplacian(float distance_i, float particle_size_i) {
        float h = particle_size_i;
        float h2 = h * h;
        if (distance_i < 0 || distance_i >= h2) return 0.0f;

        float r2 = distance_i;
        float coeff = 4.0f / (std::numbers::pi_v<float> * std::pow(h, 8));

        return coeff * -6.0f * (3.0f * h2 * h2 - 10.0f * h2 * r2 + 7.0f * r2 * r2);
    }
};

#endif
