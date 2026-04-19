#ifndef _GRAPHICS_HPP
#define _GRAPHICS_HPP

#include <stdint.h>
#include <utility>
#include <string>
#include <sycl/sycl.hpp>
#include "settings.hpp"
#include "struct.hpp"

namespace JD::graphics {

    extern uint8_t* static_rgb_buffer;
    extern int* offsets;
    extern int* cells_ctr;
    extern int* particles_loc;

    extern point* points;

    void draw_line_std_pair(uint8_t* buffer,
                            std::pair<int, int> p0,
                            std::pair<int, int> p1,
                            uint8_t r, uint8_t g, uint8_t b);

    void initBuffers();
    void drawGrid();
    void drawConnections();
    void computeStrengths();
    void initGrid();

    // assuming uint8_t
    void outputPPM(int height, int width, std::string output);
} // namespace JD::graphics

#endif
