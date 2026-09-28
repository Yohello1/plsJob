#include "render.hpp"

#include <algorithm>
#include <cmath>

namespace JD::render
{
    Rgb densityColor(float value, float maximum)
    {
        if (!std::isfinite(value) || !std::isfinite(maximum) || value <= 0.0f || maximum <= 0.0f) {
            return {0, 0, 0};
        }
        const float normalized = std::sqrt(std::clamp(value / maximum, 0.0f, 1.0f));
        const float blue = std::clamp(normalized * 4.0f, 0.0f, 1.0f);
        const float green = std::clamp((normalized - 0.25f) * 4.0f, 0.0f, 1.0f);
        const float red = std::clamp((normalized - 0.5f) * 2.0f, 0.0f, 1.0f);
        return {
            static_cast<std::uint8_t>(std::lround(red * 255.0f)),
            static_cast<std::uint8_t>(std::lround(green * 255.0f)),
            static_cast<std::uint8_t>(std::lround(blue * 255.0f)),
        };
    }

    void densityToRgb(const float* density, std::size_t count, std::uint8_t* rgb)
    {
        if (density == nullptr || rgb == nullptr) {
            return;
        }
        float maximum = 0.0f;
        for (std::size_t index = 0; index < count; ++index) {
            if (std::isfinite(density[index])) {
                maximum = std::max(maximum, density[index]);
            }
        }
        for (std::size_t index = 0; index < count; ++index) {
            const Rgb color = densityColor(density[index], maximum);
            rgb[index * 3] = color.red;
            rgb[index * 3 + 1] = color.green;
            rgb[index * 3 + 2] = color.blue;
        }
    }
}
