#ifndef _MATH_HPP
#define _MATH_HPP

#include <vector>
#include <utility>
#include <immintrin.h>
#include "struct.hpp"

namespace JD::math {

    inline int signBit(int a) {
        if (a == 0) return 0;
        return a / std::abs(a);
    }

    inline float fsignBit(float a) {
        if (a == 0.0f) return 0.0f;
        return a / std::abs(a);
    }
    inline float fdistEuclid(std::vector<float> a, std::vector<float> b) {
        float qq = 0.0f;
        // Note: Assumes a and b are the same size
        for(size_t i = 0; i < a.size(); i++) {
            qq += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(qq);
    }
    
    inline std::pair<int, int> getMidPoint(const point& p0, const point& p1) {
        return {(p0.i_x + p1.i_x) / 2, (p0.i_y + p1.i_y) / 2};
    }
    
    inline float rsqrt( float number ){
        float res;
        _mm_store_ss(&res, _mm_rsqrt_ss(_mm_set_ss(number)));
        return res;

    }

    inline float ffast_max(float a, float b) {
        return (a > b) ? a : b;
    }
} // namespace JD::math

#endif
