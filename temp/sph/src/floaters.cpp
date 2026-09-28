#include "floaters.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "graphics.hpp"
#include "sycl.hpp"

namespace JD::floaters
{
    floaters_soa floatersA{};
    block* blocks = nullptr;

    namespace
    {
        template <typename T>
        T* allocate(std::size_t count)
        {
            T* memory = ::sycl::malloc_shared<T>(count, JD::sycl::compute_queue);
            if (memory == nullptr) {
                throw std::runtime_error("SYCL allocation failed");
            }
            return memory;
        }

        bool validBox(const SpawnBox& box)
        {
            return std::isfinite(box.x) && std::isfinite(box.y) && std::isfinite(box.w) && std::isfinite(box.h) && box.w > 0.0f && box.h > 0.0f;
        }

        void resetParticle(std::size_t index, bool enabled)
        {
            floatersA.x[index] = -1000.0f;
            floatersA.y[index] = -1000.0f;
            floatersA.v_x[index] = 0.0f;
            floatersA.v_y[index] = 0.0f;
            floatersA.v_x_h[index] = 0.0f;
            floatersA.v_y_h[index] = 0.0f;
            floatersA.a_x[index] = 0.0f;
            floatersA.a_y[index] = 0.0f;
            floatersA.mass[index] = PARTICLE_MASS;
            floatersA.pressure[index] = 0.0f;
            floatersA.density[index] = PARTICLE_REFERENCE_DENSITY;
            floatersA.p_x[index] = 0.0f;
            floatersA.p_y[index] = 0.0f;
            floatersA.enabled[index] = enabled;
        }

        void spawnFluidBox(const SpawnBox& box, std::size_t target, std::size_t& current)
        {
            if (target == 0 || current >= DESIRED_FLOATERS) {
                return;
            }
            const float ratio = box.w / box.h;
            int columns = std::max(1, static_cast<int>(std::round(std::sqrt(static_cast<float>(target) * ratio))));
            int rows = static_cast<int>((target + static_cast<std::size_t>(columns) - 1) / static_cast<std::size_t>(columns));
            columns = std::max(1, columns);
            rows = std::max(1, rows);
            const float step_x = box.w / static_cast<float>(columns);
            const float step_y = box.h / static_cast<float>(rows);
            std::size_t spawned = 0;
            for (int row = 0; row < rows && spawned < target && current < DESIRED_FLOATERS; ++row) {
                for (int column = 0; column < columns && spawned < target && current < DESIRED_FLOATERS; ++column) {
                    floatersA.x[current] = box.x + (static_cast<float>(column) + 0.5f) * step_x;
                    floatersA.y[current] = box.y + (static_cast<float>(row) + 0.5f) * step_y;
                    floatersA.v_x[current] = 0.0f;
                    floatersA.v_y[current] = 0.0f;
                    floatersA.enabled[current] = true;
                    ++current;
                    ++spawned;
                }
            }
        }
    }

    void init(float spawn_x, float spawn_y, const std::vector<SpawnBox>& fluid_boxes, const std::vector<SpawnBox>& ghost_boxes)
    {
        if (floatersA.x == nullptr) {
            floatersA.density = allocate<float>(FLOATER_AMT);
            floatersA.p_x = allocate<float>(FLOATER_AMT);
            floatersA.p_y = allocate<float>(FLOATER_AMT);
            floatersA.x = allocate<float>(FLOATER_AMT);
            floatersA.y = allocate<float>(FLOATER_AMT);
            floatersA.v_x = allocate<float>(FLOATER_AMT);
            floatersA.v_y = allocate<float>(FLOATER_AMT);
            floatersA.v_x_h = allocate<float>(FLOATER_AMT);
            floatersA.v_y_h = allocate<float>(FLOATER_AMT);
            floatersA.a_x = allocate<float>(FLOATER_AMT);
            floatersA.a_y = allocate<float>(FLOATER_AMT);
            floatersA.mass = allocate<float>(FLOATER_AMT);
            floatersA.pressure = allocate<float>(FLOATER_AMT);
            floatersA.enabled = allocate<bool>(FLOATER_AMT);
        }
        if (blocks == nullptr) {
            blocks = allocate<block>(BLOCK_AMT);
        }
        initFloaters(spawn_x, spawn_y, fluid_boxes, ghost_boxes);
        initBlockRegions();
        JD::sycl::compute_queue.wait();
    }

    void initFloaters(float spawn_x, float spawn_y, const std::vector<SpawnBox>& fluid_boxes, const std::vector<SpawnBox>& ghost_boxes)
    {
        JD::sycl::compute_queue.wait();
        for (std::size_t index = 0; index < FLOATER_AMT; ++index) {
            resetParticle(index, false);
        }

        std::size_t current_fluid = 0;
        if (!fluid_boxes.empty()) {
            std::vector<std::size_t> valid_box_indices;
            for (std::size_t index = 0; index < fluid_boxes.size(); ++index) {
                if (validBox(fluid_boxes[index])) {
                    valid_box_indices.push_back(index);
                }
            }
            if (!valid_box_indices.empty()) {
                std::size_t remaining = DESIRED_FLOATERS;
                for (std::size_t valid_position = 0; valid_position < valid_box_indices.size() && remaining != 0; ++valid_position) {
                    const std::size_t boxes_left = valid_box_indices.size() - valid_position;
                    const std::size_t target = (remaining + boxes_left - 1) / boxes_left;
                    spawnFluidBox(fluid_boxes[valid_box_indices[valid_position]], target, current_fluid);
                    remaining = current_fluid < DESIRED_FLOATERS ? DESIRED_FLOATERS - current_fluid : 0;
                }
            }
        } else if (spawn_x >= 0.0f && spawn_y >= 0.0f) {
            constexpr int side = 90;
            for (std::size_t index = 0; index < DESIRED_FLOATERS; ++index) {
                floatersA.x[index] = spawn_x + static_cast<float>(index % side) * 2.1f;
                floatersA.y[index] = spawn_y + static_cast<float>(index / side) * 2.1f;
                floatersA.enabled[index] = true;
                ++current_fluid;
            }
        }

        std::size_t current_ghost = DESIRED_FLOATERS;
        for (const SpawnBox& box : ghost_boxes) {
            if (!validBox(box)) {
                continue;
            }
            for (float y = box.y; y < box.y + box.h && current_ghost < FLOATER_AMT; y += static_cast<float>(PARTICLE_GHOST_DENSITY)) {
                for (float x = box.x; x < box.x + box.w && current_ghost < FLOATER_AMT; x += static_cast<float>(PARTICLE_GHOST_DENSITY)) {
                    floatersA.x[current_ghost] = x;
                    floatersA.y[current_ghost] = y;
                    floatersA.enabled[current_ghost] = false;
                    ++current_ghost;
                }
            }
        }

        int shell = 0;
        while (current_ghost < FLOATER_AMT) {
            const int x0 = BUFFER_PADDING - shell;
            const int y0 = BUFFER_PADDING - shell;
            const int x1 = BUFFER_PADDING + BUFFER_WORKING + shell;
            const int y1 = BUFFER_PADDING + BUFFER_WORKING + shell;
            for (int x = x0; x < x1 && current_ghost < FLOATER_AMT; x += PARTICLE_GHOST_DENSITY) {
                floatersA.x[current_ghost] = static_cast<float>(x);
                floatersA.y[current_ghost] = static_cast<float>(y0);
                ++current_ghost;
            }
            for (int y = y0; y < y1 && current_ghost < FLOATER_AMT; y += PARTICLE_GHOST_DENSITY) {
                floatersA.x[current_ghost] = static_cast<float>(x1);
                floatersA.y[current_ghost] = static_cast<float>(y);
                ++current_ghost;
            }
            for (int x = x1; x > x0 && current_ghost < FLOATER_AMT; x -= PARTICLE_GHOST_DENSITY) {
                floatersA.x[current_ghost] = static_cast<float>(x);
                floatersA.y[current_ghost] = static_cast<float>(y1);
                ++current_ghost;
            }
            for (int y = y1; y > y0 && current_ghost < FLOATER_AMT; y -= PARTICLE_GHOST_DENSITY) {
                floatersA.x[current_ghost] = static_cast<float>(x0);
                floatersA.y[current_ghost] = static_cast<float>(y);
                ++current_ghost;
            }
            ++shell;
            if (x0 <= 0 || y0 <= 0) {
                break;
            }
        }
    }

    void drawFloaters()
    {
        if (JD::graphics::static_rgb_buffer == nullptr) {
            return;
        }
        for (std::size_t index = 0; index < FLOATER_AMT; ++index) {
            if (!std::isfinite(floatersA.x[index]) || !std::isfinite(floatersA.y[index]) || floatersA.x[index] < 0.0f || floatersA.y[index] < 0.0f || floatersA.x[index] >= BUFFER_WIDTH || floatersA.y[index] >= BUFFER_HEIGHT) {
                continue;
            }
            const int px = static_cast<int>(floatersA.x[index]);
            const int py = static_cast<int>(floatersA.y[index]);
            if (px < 0 || px >= BUFFER_WIDTH || py < 0 || py >= BUFFER_HEIGHT) {
                continue;
            }
            const std::size_t index_buffer = static_cast<std::size_t>(px) * BYTES_PER_PIXEL + static_cast<std::size_t>(py) * BUFFER_WIDTH * BYTES_PER_PIXEL;
            const int channel = floatersA.enabled[index] ? 0 : 1;
            JD::graphics::static_rgb_buffer[index_buffer + static_cast<std::size_t>(channel)] = 250;
        }
    }

    void initBlockRegions()
    {
        if (blocks == nullptr) {
            return;
        }
        const int width = BUFFER_LINE;
        const int height = static_cast<int>(BLOCK_AMT / static_cast<std::size_t>(BUFFER_LINE));
        for (std::size_t index = 0; index < BLOCK_AMT; ++index) {
            const int row = static_cast<int>(index / static_cast<std::size_t>(width));
            const int column = static_cast<int>(index % static_cast<std::size_t>(width));
            for (int region = 0; region < BLOCK_NEIGHBOR_COUNT; ++region) {
                const int dy = region / BLOCK_NEIGHBOR_DIM - INFLUENCE_RADIUS;
                const int dx = region % BLOCK_NEIGHBOR_DIM - INFLUENCE_RADIUS;
                const int target_row = row + dy;
                const int target_column = column + dx;
                if (target_row < 0 || target_row >= height || target_column < 0 || target_column >= width) {
                    blocks[index].regions[region] = std::numeric_limits<std::uint32_t>::max();
                } else {
                    blocks[index].regions[region] = static_cast<std::uint32_t>(target_row * width + target_column);
                }
            }
        }
    }

    void shutdown()
    {
        JD::sycl::compute_queue.wait();
        if (floatersA.density != nullptr) ::sycl::free(floatersA.density, JD::sycl::compute_queue);
        if (floatersA.p_x != nullptr) ::sycl::free(floatersA.p_x, JD::sycl::compute_queue);
        if (floatersA.p_y != nullptr) ::sycl::free(floatersA.p_y, JD::sycl::compute_queue);
        if (floatersA.x != nullptr) ::sycl::free(floatersA.x, JD::sycl::compute_queue);
        if (floatersA.y != nullptr) ::sycl::free(floatersA.y, JD::sycl::compute_queue);
        if (floatersA.v_x != nullptr) ::sycl::free(floatersA.v_x, JD::sycl::compute_queue);
        if (floatersA.v_y != nullptr) ::sycl::free(floatersA.v_y, JD::sycl::compute_queue);
        if (floatersA.v_x_h != nullptr) ::sycl::free(floatersA.v_x_h, JD::sycl::compute_queue);
        if (floatersA.v_y_h != nullptr) ::sycl::free(floatersA.v_y_h, JD::sycl::compute_queue);
        if (floatersA.a_x != nullptr) ::sycl::free(floatersA.a_x, JD::sycl::compute_queue);
        if (floatersA.a_y != nullptr) ::sycl::free(floatersA.a_y, JD::sycl::compute_queue);
        if (floatersA.mass != nullptr) ::sycl::free(floatersA.mass, JD::sycl::compute_queue);
        if (floatersA.pressure != nullptr) ::sycl::free(floatersA.pressure, JD::sycl::compute_queue);
        if (floatersA.enabled != nullptr) ::sycl::free(floatersA.enabled, JD::sycl::compute_queue);
        floatersA = {};
        if (blocks != nullptr) {
            ::sycl::free(blocks, JD::sycl::compute_queue);
            blocks = nullptr;
        }
    }
}
