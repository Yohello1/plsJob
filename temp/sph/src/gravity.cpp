#include "gravity.hpp"

#include "settings.hpp"

namespace JD::gravity
{
    float gravityAcceleration()
    {
        return PARTICLE_MASS * PARTICLE_GRAVITY;
    }
}
