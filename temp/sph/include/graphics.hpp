#ifndef JD_GRAPHICS_HPP
#define JD_GRAPHICS_HPP

#include <cstdint>
#include <string>
#include <utility>

#include "settings.hpp"
#include "struct.hpp"

namespace JD::graphics
{
    extern std::uint8_t* static_rgb_buffer;
    extern int* offsets;
    extern int* cells_ctr;
    extern int* particles_loc;
    extern point* points;

    void draw_line_std_pair(std::uint8_t* buffer,
                            std::pair<int, int> p0,
                            std::pair<int, int> p1,
                            std::uint8_t r,
                            std::uint8_t g,
                            std::uint8_t b);
    void initBuffers();
    void InitializeStaticBuffer();
    void drawGrid();
    void drawConnections();
    void drawDensity(const float* density);
    void computeStrengths();
    void initGrid();
    void outputPPM(int height, int width, const std::string& output);
    void shutdown();
}

#endif
