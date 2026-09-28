#include "floaters.hpp"
#include "graphics.hpp"
#include "math.hpp"
#include "render.hpp"
#include "spatial.hpp"
#include "sycl.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace JD::graphics
{
    std::uint8_t* static_rgb_buffer = nullptr;
    int* offsets = nullptr;
    int* cells_ctr = nullptr;
    int* particles_loc = nullptr;
    point* points = nullptr;

    void initBuffers()
    {
        if (static_rgb_buffer == nullptr) {
            static_rgb_buffer = ::sycl::malloc_shared<std::uint8_t>(static_cast<std::size_t>(BUFFER_WIDTH) * BUFFER_HEIGHT * BYTES_PER_PIXEL, JD::sycl::compute_queue);
        }
        if (offsets == nullptr) {
            offsets = ::sycl::malloc_shared<int>(static_cast<std::size_t>(BUFFER_LINE) * BUFFER_LINE, JD::sycl::compute_queue);
        }
        if (cells_ctr == nullptr) {
            cells_ctr = ::sycl::malloc_shared<int>(static_cast<std::size_t>(BUFFER_LINE) * BUFFER_LINE, JD::sycl::compute_queue);
        }
        if (particles_loc == nullptr) {
            particles_loc = ::sycl::malloc_shared<int>(JD::floaters::FLOATER_AMT, JD::sycl::compute_queue);
        }
        if (static_rgb_buffer == nullptr || offsets == nullptr || cells_ctr == nullptr || particles_loc == nullptr) {
            throw std::runtime_error("SYCL graphics allocation failed");
        }
        JD::sycl::compute_queue.fill(static_rgb_buffer, static_cast<std::uint8_t>(0), static_cast<std::size_t>(BUFFER_WIDTH) * BUFFER_HEIGHT * BYTES_PER_PIXEL).wait();
        JD::sycl::compute_queue.fill(offsets, 0, static_cast<std::size_t>(BUFFER_LINE) * BUFFER_LINE).wait();
        JD::sycl::compute_queue.fill(cells_ctr, 0, static_cast<std::size_t>(BUFFER_LINE) * BUFFER_LINE).wait();
        JD::sycl::compute_queue.fill(particles_loc, 0, JD::floaters::FLOATER_AMT).wait();
    }

    void InitializeStaticBuffer()
    {
        initBuffers();
    }

    void draw_line_std_pair(std::uint8_t* buffer,
                            std::pair<int, int> p0,
                            std::pair<int, int> p1,
                            std::uint8_t r,
                            std::uint8_t g,
                            std::uint8_t b)
    {
        if (buffer == nullptr) {
            return;
        }
        int x0 = p0.first;
        int y0 = p0.second;
        const int x1 = p1.first;
        const int y1 = p1.second;
        const int dx = std::abs(x1 - x0);
        const int dy = std::abs(y1 - y0);
        const int sx = x0 < x1 ? 1 : -1;
        const int sy = y0 < y1 ? 1 : -1;
        int error = (dx > dy ? dx : -dy) / 2;
        while (true) {
            if (x0 >= 0 && x0 < BUFFER_WIDTH && y0 >= 0 && y0 < BUFFER_HEIGHT) {
                const std::size_t index = (static_cast<std::size_t>(y0) * BUFFER_WIDTH + static_cast<std::size_t>(x0)) * BYTES_PER_PIXEL;
                buffer[index] = r;
                buffer[index + 1] = g;
                buffer[index + 2] = b;
            }
            if (x0 == x1 && y0 == y1) {
                break;
            }
            const int twice_error = 2 * error;
            if (twice_error > -dy) {
                error -= dy;
                x0 += sx;
            }
            if (twice_error < dx) {
                error += dx;
                y0 += sy;
            }
        }
    }

    void initGrid()
    {
        if (points == nullptr) {
            points = new point[POINTS_AMT];
        }
        std::vector<std::pair<int, int>> regions = JD::spatial::calculateRegionsOffsets();
        std::sort(regions.begin(), regions.end(), [](const auto& left, const auto& right) {
            return left.first + left.second * BUFFER_LINE < right.first + right.second * BUFFER_LINE;
        });
        for (int index = 0; index < POINTS_AMT; ++index) {
            const int x = index % POINTS_WIDTH;
            const int y = index / POINTS_WIDTH;
            points[index].x = static_cast<std::uint16_t>(x);
            points[index].y = static_cast<std::uint16_t>(y);
            points[index].i_x = static_cast<std::uint16_t>(x * DISTANCE_BETWEEN_POINTS + BUFFER_PADDING);
            points[index].i_y = static_cast<std::uint16_t>(y * DISTANCE_BETWEEN_POINTS + BUFFER_PADDING);
            points[index].id = static_cast<std::uint16_t>(index);
            points[index].strength = 0.0f;
            const int base_x = points[index].i_x / DISTANCE_BETWEEN_POINTS;
            const int base_y = points[index].i_y / DISTANCE_BETWEEN_POINTS;
            for (int region = 0; region < REGIONS_AMT; ++region) {
                const int target_x = base_x + regions[static_cast<std::size_t>(region)].first;
                const int target_y = base_y + regions[static_cast<std::size_t>(region)].second;
                points[index].regions[region] = target_x >= 0 && target_x < BUFFER_LINE && target_y >= 0 && target_y < BUFFER_LINE ? target_x + target_y * BUFFER_LINE : -1;
            }
        }
    }

    void computeStrengths()
    {
        if (points == nullptr || offsets == nullptr || cells_ctr == nullptr || particles_loc == nullptr) {
            return;
        }
        for (int index = 0; index < POINTS_AMT; ++index) {
            float strength = 0.0f;
            for (int region = 0; region < REGIONS_AMT; ++region) {
                const int cell = points[index].regions[region];
                if (cell < 0) {
                    continue;
                }
                const int start = offsets[cell];
                const int count = cells_ctr[cell];
                for (int item = 0; item < count; ++item) {
                    const int particle = particles_loc[start + item];
                    const float dx = JD::floaters::floatersA.x[particle] - points[index].i_x;
                    const float dy = JD::floaters::floatersA.y[particle] - points[index].i_y;
                    const float distance_squared = std::max(0.001f, dx * dx + dy * dy);
                    strength += JD::floaters::floatersA.density[particle] / distance_squared;
                }
            }
            points[index].strength = strength;
        }
    }

    void drawConnections()
    {
        if (points == nullptr || static_rgb_buffer == nullptr) {
            return;
        }
        constexpr int lut[16][4] = {
            {-1, -1, -1, -1}, {0, 3, -1, -1}, {0, 1, -1, -1}, {3, 1, -1, -1},
            {1, 2, -1, -1}, {0, 1, 2, 3}, {0, 2, -1, -1}, {3, 2, -1, -1},
            {3, 2, -1, -1}, {0, 2, -1, -1}, {0, 3, 1, 2}, {1, 2, -1, -1},
            {3, 1, -1, -1}, {0, 1, -1, -1}, {0, 3, -1, -1}, {-1, -1, -1, -1}
        };
        for (int row = 0; row + 1 < POINTS_HEIGHT; ++row) {
            for (int column = 0; column + 1 < POINTS_WIDTH; ++column) {
                const int top_left = row * POINTS_WIDTH + column;
                const int top_right = top_left + 1;
                const int bottom_left = top_left + POINTS_WIDTH;
                const int bottom_right = bottom_left + 1;
                const int configuration = (points[top_left].strength >= THRESHOLD)
                    | ((points[top_right].strength >= THRESHOLD) << 1)
                    | ((points[bottom_right].strength >= THRESHOLD) << 2)
                    | ((points[bottom_left].strength >= THRESHOLD) << 3);
                if (configuration == 0 || configuration == 15) {
                    continue;
                }
                const std::pair<int, int> corners[4] = {
                    JD::math::getMidPoint(points[top_left], points[top_right]),
                    JD::math::getMidPoint(points[top_right], points[bottom_right]),
                    JD::math::getMidPoint(points[bottom_right], points[bottom_left]),
                    JD::math::getMidPoint(points[bottom_left], points[top_left])
                };
                const int* edges = lut[configuration];
                draw_line_std_pair(static_rgb_buffer, corners[edges[0]], corners[edges[1]], 255, 255, 255);
                if (edges[2] >= 0) {
                    draw_line_std_pair(static_rgb_buffer, corners[edges[2]], corners[edges[3]], 255, 255, 255);
                }
            }
        }
    }

    void drawGrid()
    {
        if (points == nullptr || static_rgb_buffer == nullptr) {
            return;
        }
        for (int index = 0; index < POINTS_AMT; ++index) {
            if (points[index].strength < THRESHOLD) {
                continue;
            }
            const std::size_t offset = (static_cast<std::size_t>(points[index].i_y) * BUFFER_WIDTH + points[index].i_x) * BYTES_PER_PIXEL;
            static_rgb_buffer[offset] = 255;
            static_rgb_buffer[offset + 1] = 255;
            static_rgb_buffer[offset + 2] = 255;
        }
    }

    void drawDensity(const float* density)
    {
        if (static_rgb_buffer == nullptr || density == nullptr) {
            return;
        }
        JD::sycl::compute_queue.wait();
        constexpr std::size_t pixel_count = static_cast<std::size_t>(BUFFER_WIDTH) * BUFFER_HEIGHT;
        JD::render::densityToRgb(density, pixel_count, static_rgb_buffer);
    }

    void outputPPM(int height, int width, const std::string& output)
    {
        if (static_rgb_buffer == nullptr || width <= 0 || height <= 0) {
            return;
        }
        std::ofstream file(output, std::ios::binary);
        if (!file) {
            return;
        }
        file << "P6\n" << width << ' ' << height << "\n255\n";
        file.write(reinterpret_cast<const char*>(static_rgb_buffer), static_cast<std::streamsize>(width) * height * BYTES_PER_PIXEL);
    }

    void shutdown()
    {
        JD::sycl::compute_queue.wait();
        if (static_rgb_buffer != nullptr) ::sycl::free(static_rgb_buffer, JD::sycl::compute_queue);
        if (offsets != nullptr) ::sycl::free(offsets, JD::sycl::compute_queue);
        if (cells_ctr != nullptr) ::sycl::free(cells_ctr, JD::sycl::compute_queue);
        if (particles_loc != nullptr) ::sycl::free(particles_loc, JD::sycl::compute_queue);
        static_rgb_buffer = nullptr;
        offsets = nullptr;
        cells_ctr = nullptr;
        particles_loc = nullptr;
        delete[] points;
        points = nullptr;
    }
}
