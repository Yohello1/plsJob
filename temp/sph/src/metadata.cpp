#include "metadata.hpp"

#include <fstream>
#include <sstream>

namespace JD::metadata
{
    const char* modelVariant(Variant variant)
    {
        return variant == Variant::DensityVelocity ? "density-velocity" : "density-only";
    }

    std::string makeJson(Variant variant, int width, int height)
    {
        std::ostringstream output;
        output << "{\n"
               << "  \"model_variant\": \"" << modelVariant(variant) << "\",\n"
               << "  \"width\": " << width << ",\n"
               << "  \"height\": " << height << ",\n"
               << "  \"fields\": [\"density\", \"velocity_x\", \"velocity_y\", \"obstacle_mask\"],\n"
               << "  \"dtype\": \"" << dtype << "\"\n"
               << "}\n";
        return output.str();
    }

    bool write(const std::filesystem::path& path, Variant variant, int width, int height)
    {
        if (path.empty() || width <= 0 || height <= 0) {
            return false;
        }
        std::error_code error;
        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(path.parent_path(), error);
            if (error) {
                return false;
            }
        }
        std::ofstream output(path);
        if (!output) {
            return false;
        }
        output << makeJson(variant, width, height);
        return static_cast<bool>(output);
    }
}
