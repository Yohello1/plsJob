#include "math.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace JD::math
{
    int signBit(int value)
    {
        if (value == 0) {
            return 0;
        }
        return value > 0 ? 1 : -1;
    }

    float fsignBit(float value)
    {
        if (value == 0.0f) {
            return 0.0f;
        }
        return value > 0.0f ? 1.0f : -1.0f;
    }

    float fdistEuclid(const std::vector<float>& a, const std::vector<float>& b)
    {
        const std::size_t count = std::min(a.size(), b.size());
        float squared = 0.0f;
        for (std::size_t index = 0; index < count; ++index) {
            const float difference = a[index] - b[index];
            squared += difference * difference;
        }
        return std::sqrt(squared);
    }

    std::pair<int, int> getMidPoint(const point& p0, const point& p1)
    {
        return {(p0.i_x + p1.i_x) / 2, (p0.i_y + p1.i_y) / 2};
    }

    float rsqrt(float value)
    {
        if (value <= 0.0f) {
            return 0.0f;
        }
        return 1.0f / std::sqrt(value);
    }
}
