#include "cli.hpp"
#include "math.hpp"
#include "metadata.hpp"
#include "render.hpp"
#include "settings.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace
{
    int failures = 0;

    void check(bool condition, const std::string& name)
    {
        if (!condition) {
            std::cerr << "FAIL: " << name << '\n';
            ++failures;
        }
    }

    bool has(const std::string& text, const std::string& value)
    {
        return text.find(value) != std::string::npos;
    }
}

int main()
{
    static_assert(JD::metadata::field_count == 4);
    static_assert(BUFFER_WIDTH == 400);
    static_assert(BUFFER_HEIGHT == 400);
    static_assert(JD::metadata::frameBytes(BUFFER_WIDTH, BUFFER_HEIGHT) == 2560000);

    const JD::cli::Result defaults = JD::cli::parse(std::vector<std::string>{});
    check(defaults.ok, "default arguments parse");
    check(defaults.options.frame_count == 1, "default frame count is positive");
    check(!defaults.options.headless, "default is not headless");
    check(defaults.options.render_mode == JD::cli::RenderMode::Particles, "default render mode is particles");

    const JD::cli::Result density_render = JD::cli::parse({"--headless", "--render", "density"});
    check(density_render.ok && density_render.options.render_mode == JD::cli::RenderMode::Density, "density render parses");
    const JD::cli::Result density_shorthand = JD::cli::parse({"--render-density"});
    check(density_shorthand.ok && density_shorthand.options.render_mode == JD::cli::RenderMode::Density, "density shorthand parses");
    check(!JD::cli::parse({"--render", "unknown"}).ok, "unknown render mode rejected");

    const JD::render::Rgb empty_color = JD::render::densityColor(0.0f, 1.0f);
    check(empty_color.red == 0 && empty_color.green == 0 && empty_color.blue == 0, "zero density is black");
    const JD::render::Rgb peak_color = JD::render::densityColor(1.0f, 1.0f);
    check(peak_color.red == 255 && peak_color.green == 255 && peak_color.blue == 255, "maximum density is white");
    const JD::render::Rgb invalid_color = JD::render::densityColor(-1.0f, 1.0f);
    check(invalid_color.red == 0 && invalid_color.green == 0 && invalid_color.blue == 0, "invalid density is black");
    const float density_values[]{0.0f, 1.0f};
    std::uint8_t density_rgb[6]{};
    JD::render::densityToRgb(density_values, 2, density_rgb);
    check(density_rgb[0] == 0 && density_rgb[1] == 0 && density_rgb[2] == 0, "density frame background is black");
    check(density_rgb[3] == 255 && density_rgb[4] == 255 && density_rgb[5] == 255, "density frame peak is white");

    const JD::cli::Result configured = JD::cli::parse({"--headless", "4", "--fluid", "10", "20", "30", "40", "-g", "50", "60", "70", "80"});
    check(configured.ok, "mixed options parse");
    check(configured.options.frame_count == 4, "positional frame count parses after options");
    check(configured.options.headless, "headless parses");
    check(configured.options.fluid_boxes.size() == 1, "fluid box parses");
    check(configured.options.ghost_boxes.size() == 1, "ghost box parses");
    check(configured.options.fluid_boxes[0].w == 30.0f, "fluid width parses");
    check(configured.options.ghost_boxes[0].h == 80.0f, "ghost height parses");

    const JD::cli::Result named_frames = JD::cli::parse({"--frames", "2", "--headless"});
    check(named_frames.ok && named_frames.options.frame_count == 2, "named frame count parses");

    check(!JD::cli::parse({"0"}).ok, "zero frame count rejected");
    check(!JD::cli::parse({"--frames", "-2"}).ok, "negative frame count rejected");
    check(!JD::cli::parse({"--fluid", "1", "2", "3"}).ok, "short fluid box rejected");
    check(!JD::cli::parse({"--fluid", "1", "2", "0", "4"}).ok, "nonpositive box rejected");
    check(!JD::cli::parse({"--unknown"}).ok, "unknown option rejected");
    check(!JD::cli::parse({"2", "3"}).ok, "multiple frame counts rejected");

    const std::string density_json = JD::metadata::makeJson(JD::metadata::Variant::DensityOnly, 400, 400);
    check(has(density_json, "\"model_variant\": \"density-only\""), "density metadata variant");
    check(has(density_json, "\"width\": 400"), "metadata width");
    check(has(density_json, "\"height\": 400"), "metadata height");
    check(has(density_json, "\"fields\": [\"density\", \"velocity_x\", \"velocity_y\", \"obstacle_mask\"]"), "metadata fields");
    check(has(density_json, "\"dtype\": \"float32\""), "metadata dtype");

    const std::string velocity_json = JD::metadata::makeJson(JD::metadata::Variant::DensityVelocity, 8, 6);
    check(has(velocity_json, "\"model_variant\": \"density-velocity\""), "velocity metadata variant");

    const auto path = std::filesystem::temp_directory_path() / "pls_old_sph_metadata_test.json";
    std::filesystem::remove(path);
    check(JD::metadata::write(path, JD::metadata::Variant::DensityOnly, 8, 6), "metadata file writes");
    std::ifstream input(path);
    const std::string written((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    check(has(written, "\"dtype\": \"float32\""), "metadata file content");
    std::filesystem::remove(path);

    check(JD::math::signBit(-4) == -1, "integer sign helper");
    check(JD::math::fsignBit(0.0f) == 0.0f, "float sign helper at zero");
    check(JD::math::fdistEuclid({0.0f, 0.0f}, {3.0f, 4.0f}) == 5.0f, "distance helper");

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "core tests passed\n";
    return 0;
}
