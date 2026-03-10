#include "math.hpp"

#include <immintrin.h>
#include <cmath>
#include <vector>

namespace JD::math {

    int signBit(int a) {
        if (a == 0) return 0;
        return a / std::abs(a);
    }

    float fsignBit(float a) {
        if (a == 0.0f) return 0.0f;
        return a / std::abs(a);
    }

    float fdistEuclid(std::vector<float> a, std::vector<float> b) {
        float qq = 0.0f;
        // Note: Assumes a and b are the same size
        for(size_t i = 0; i < a.size(); i++) {
            qq += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(qq);
    }

    std::pair<int, int> getMidPoint(const point& p0, const point& p1) {
        return {(p0.i_x + p1.i_x) / 2, (p0.i_y + p1.i_y) / 2};
    }

    // Source - https://stackoverflow.com/a/53893163
    // Posted by user2344271
    // Retrieved 2026-02-12, License - CC BY-SA 4.0
    float rsqrt( float number ){
        float res;
        _mm_store_ss(&res, _mm_rsqrt_ss(_mm_set_ss(number)));
        return res;

    }

    float ffast_max(float a, float b) {
        return (a > b) ? a : b;
    }


} // namespace JD::math
