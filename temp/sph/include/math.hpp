#ifndef JD_MATH_HPP
#define JD_MATH_HPP

#include <utility>
#include <vector>

#include "struct.hpp"

namespace JD::math
{
    int signBit(int value);
    float fsignBit(float value);
    float fdistEuclid(const std::vector<float>& a, const std::vector<float>& b);
    std::pair<int, int> getMidPoint(const point& p0, const point& p1);
    float rsqrt(float value);
    inline float ffast_max(float left, float right)
    {
        return left > right ? left : right;
    }
}

#endif
