#include <numbers>

#include "struct.hpp"
#include "spiky_k.hpp"
#include "math.hpp"

void JD::Spiky_k::gradient(float dx, float dy, float dist_sq, float h, force& out_force) {
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
