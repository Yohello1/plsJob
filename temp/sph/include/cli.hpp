#ifndef JD_CLI_HPP
#define JD_CLI_HPP

#include <string>
#include <vector>

#include "struct.hpp"

namespace JD::cli
{
    inline constexpr int default_frame_count = 1;

    enum class RenderMode
    {
        Particles,
        Density
    };

    struct Options
    {
        int frame_count = default_frame_count;
        bool headless = false;
        bool help = false;
        RenderMode render_mode = RenderMode::Particles;
        std::vector<SpawnBox> fluid_boxes;
        std::vector<SpawnBox> ghost_boxes;
    };

    struct Result
    {
        bool ok = true;
        std::string error;
        Options options;
    };

    Result parse(int argc, char* const argv[]);
    Result parse(const std::vector<std::string>& arguments);
    std::string usage();
}

#endif
