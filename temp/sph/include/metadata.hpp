#ifndef JD_METADATA_HPP
#define JD_METADATA_HPP

#include <array>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <string>

namespace JD::metadata
{
    enum class Variant
    {
        DensityOnly,
        DensityVelocity
    };

    inline constexpr std::array<const char*, 4> field_names{{"density", "velocity_x", "velocity_y", "obstacle_mask"}};
    inline constexpr int field_count = 4;
    inline constexpr const char* dtype = "float32";
    inline constexpr const char* filename = "metadata.json";

    const char* modelVariant(Variant variant);
    inline constexpr std::size_t frameBytes(int width, int height)
    {
        if (width <= 0 || height <= 0) {
            return 0;
        }
        const std::size_t w = static_cast<std::size_t>(width);
        const std::size_t h = static_cast<std::size_t>(height);
        if (w > std::numeric_limits<std::size_t>::max() / h) {
            return 0;
        }
        const std::size_t pixels = w * h;
        const std::size_t bytes_per_pixel = static_cast<std::size_t>(field_count) * sizeof(float);
        if (pixels > std::numeric_limits<std::size_t>::max() / bytes_per_pixel) {
            return 0;
        }
        return pixels * bytes_per_pixel;
    }
    std::string makeJson(Variant variant, int width, int height);
    bool write(const std::filesystem::path& path, Variant variant, int width, int height);
}

#endif
