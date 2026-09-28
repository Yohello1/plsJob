#ifndef JD_POLY6_HPP
#define JD_POLY6_HPP

#include "settings.hpp"
#include "struct.hpp"

namespace JD::Poly6_k
{
    inline float smoothing(float distance_squared, float particle_size)
    {
        const float h2 = particle_size * particle_size;
        if (distance_squared <= 0.0f || distance_squared >= h2) {
            return 0.0f;
        }
        const float h3 = h2 * particle_size;
        const float h9 = h3 * h3 * h3;
        const float difference = h2 - distance_squared;
        const float coefficient = 315.0f / (64.0f * PI * h9);
        return coefficient * difference * difference * difference;
    }

    inline void gradient(float dx, float dy, float distance_squared, float particle_size, force& result)
    {
        const float h2 = particle_size * particle_size;
        if (distance_squared <= 0.0f || distance_squared >= h2) {
            result = {0.0f, 0.0f};
            return;
        }
        const float difference = h2 - distance_squared;
        const float coefficient = 4.0f / (PI * particle_size * particle_size * particle_size * particle_size * particle_size * particle_size * particle_size * particle_size);
        const float scalar = coefficient * -6.0f * difference * difference;
        result.x = scalar * dx;
        result.y = scalar * dy;
    }

    inline float laplacian(float distance_squared, float particle_size)
    {
        const float h2 = particle_size * particle_size;
        if (distance_squared <= 0.0f || distance_squared >= h2) {
            return 0.0f;
        }
        const float coefficient = 4.0f / (PI * particle_size * particle_size * particle_size * particle_size * particle_size * particle_size * particle_size * particle_size);
        return coefficient * -6.0f * (3.0f * h2 * h2 - 10.0f * h2 * distance_squared + 7.0f * distance_squared * distance_squared);
    }
}

#endif
