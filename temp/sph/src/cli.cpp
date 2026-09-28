#include "cli.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>

namespace JD::cli
{
    namespace
    {
        bool parsePositiveInt(const std::string& text, int& value)
        {
            if (text.empty()) {
                return false;
            }
            errno = 0;
            char* end = nullptr;
            const long long parsed = std::strtoll(text.c_str(), &end, 10);
            if (errno == ERANGE || end == text.c_str() || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
                return false;
            }
            value = static_cast<int>(parsed);
            return true;
        }

        bool parseFiniteFloat(const std::string& text, float& value)
        {
            if (text.empty()) {
                return false;
            }
            errno = 0;
            char* end = nullptr;
            const float parsed = std::strtof(text.c_str(), &end);
            if (errno == ERANGE || end == text.c_str() || *end != '\0' || !std::isfinite(parsed)) {
                return false;
            }
            value = parsed;
            return true;
        }

        bool parseBox(const std::vector<std::string>& arguments,
                      std::size_t& index,
                      SpawnBox& box,
                      std::string& error)
        {
            if (index + 4 >= arguments.size()) {
                error = "box options require x y width height";
                return false;
            }
            float values[4]{};
            for (int value_index = 0; value_index < 4; ++value_index) {
                if (!parseFiniteFloat(arguments[index + static_cast<std::size_t>(value_index) + 1], values[value_index])) {
                    error = "box coordinates and dimensions must be finite numbers";
                    return false;
                }
            }
            if (values[2] <= 0.0f || values[3] <= 0.0f) {
                error = "box width and height must be positive";
                return false;
            }
            box = {values[0], values[1], values[2], values[3]};
            index += 4;
            return true;
        }

        Result failure(std::string message)
        {
            Result result;
            result.ok = false;
            result.error = std::move(message);
            return result;
        }
    }

    Result parse(int argc, char* const argv[])
    {
        std::vector<std::string> arguments;
        if (argc > 1 && argv != nullptr) {
            arguments.reserve(static_cast<std::size_t>(argc - 1));
            for (int index = 1; index < argc; ++index) {
                if (argv[index] == nullptr) {
                    return failure("null argument");
                }
                arguments.emplace_back(argv[index]);
            }
        }
        return parse(arguments);
    }

    Result parse(const std::vector<std::string>& arguments)
    {
        Result result;
        bool frame_seen = false;
        for (std::size_t index = 0; index < arguments.size(); ++index) {
            const std::string& argument = arguments[index];
            if (argument == "--headless") {
                result.options.headless = true;
            } else if (argument == "--help" || argument == "-h") {
                result.options.help = true;
            } else if (argument == "--render-density") {
                result.options.render_mode = RenderMode::Density;
            } else if (argument == "--render") {
                if (index + 1 >= arguments.size()) {
                    return failure("--render requires particles or density");
                }
                const std::string& mode = arguments[++index];
                if (mode == "particles") {
                    result.options.render_mode = RenderMode::Particles;
                } else if (mode == "density") {
                    result.options.render_mode = RenderMode::Density;
                } else {
                    return failure("--render requires particles or density");
                }
            } else if (argument == "--render-density" || argument.rfind("--render=", 0) == 0) {
                const std::size_t separator = argument.find('=');
                if (separator == std::string::npos) {
                    result.options.render_mode = RenderMode::Density;
                } else if (argument.substr(separator + 1) == "density") {
                    result.options.render_mode = RenderMode::Density;
                } else {
                    return failure("--render requires particles or density");
                }
            } else if (argument == "--fluid" || argument == "-f" || argument == "--ghost" || argument == "-g") {
                SpawnBox box{};
                std::string error;
                if (!parseBox(arguments, index, box, error)) {
                    return failure(error);
                }
                if (argument == "--fluid" || argument == "-f") {
                    result.options.fluid_boxes.push_back(box);
                } else {
                    result.options.ghost_boxes.push_back(box);
                }
            } else if (argument == "--frames") {
                if (index + 1 >= arguments.size() || !parsePositiveInt(arguments[index + 1], result.options.frame_count)) {
                    return failure("--frames requires a positive integer");
                }
                frame_seen = true;
                ++index;
            } else if (!argument.empty() && argument.front() == '-') {
                return failure("unknown option: " + argument);
            } else {
                if (frame_seen || !parsePositiveInt(argument, result.options.frame_count)) {
                    return failure("frame count must be one positive integer");
                }
                frame_seen = true;
            }
        }
        return result;
    }

    std::string usage()
    {
        return "Usage: draw2 [frame_count] [options]\n"
               "       draw2 --frames frame_count [options]\n"
               "Options:\n"
               "  --headless              run without an SDL window\n"
               "  --render mode           render particles or density\n"
               "  --render-density        shorthand for --render density\n"
               "  --fluid x y w h         add a fluid spawn box\n"
               "  --ghost x y w h         add a static obstacle box\n"
               "  -f x y w h              alias for --fluid\n"
               "  -g x y w h              alias for --ghost\n"
               "  --frames n              set a positive frame count\n"
               "  -h, --help              show this help\n";
    }
}
