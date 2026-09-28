#ifndef JD_RENDER_HPP
#define JD_RENDER_HPP

#include <cstddef>
#include <cstdint>

namespace JD::render
{
    struct Rgb
    {
        std::uint8_t red;
        std::uint8_t green;
        std::uint8_t blue;
    };

    Rgb densityColor(float value, float maximum);
    void densityToRgb(const float* density, std::size_t count, std::uint8_t* rgb);
}

#endif
