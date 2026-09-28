#include "spatial.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "floaters.hpp"
#include "graphics.hpp"
#include "settings.hpp"
#include "sycl.hpp"

namespace JD::spatial
{
    namespace
    {
        std::array<int, static_cast<std::size_t>(BUFFER_LINE) * BUFFER_LINE> current_positions{};

        void waitForQueue()
        {
            JD::sycl::compute_queue.wait();
        }
    }

    std::optional<std::size_t> cellIndex(float x, float y)
    {
        if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0f || y < 0.0f || x >= BUFFER_WIDTH || y >= BUFFER_HEIGHT) {
            return std::nullopt;
        }
        const int gx = static_cast<int>(x / DISTANCE_BETWEEN_POINTS);
        const int gy = static_cast<int>(y / DISTANCE_BETWEEN_POINTS);
        if (gx < 0 || gx >= BUFFER_LINE || gy < 0 || gy >= BUFFER_LINE) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(gx + gy * BUFFER_LINE);
    }

    void offsetsCreation()
    {
        if (JD::graphics::cells_ctr == nullptr || JD::graphics::offsets == nullptr) {
            return;
        }
        waitForQueue();
        std::fill_n(JD::graphics::cells_ctr, static_cast<std::size_t>(BUFFER_LINE) * BUFFER_LINE, 0);
        for (std::size_t index = 0; index < JD::floaters::FLOATER_AMT; ++index) {
            const auto cell = cellIndex(JD::floaters::floatersA.x[index], JD::floaters::floatersA.y[index]);
            if (cell.has_value()) {
                ++JD::graphics::cells_ctr[*cell];
            }
        }
        int offset = 0;
        for (std::size_t index = 0; index < static_cast<std::size_t>(BUFFER_LINE) * BUFFER_LINE; ++index) {
            JD::graphics::offsets[index] = offset;
            offset += JD::graphics::cells_ctr[index];
        }
    }

    std::vector<std::pair<int, int>> calculateRegionsOffsets()
    {
        std::vector<std::pair<int, int>> result;
        for (int y = -INFLUENCE_RADIUS; y <= INFLUENCE_RADIUS; ++y) {
            for (int x = -INFLUENCE_RADIUS; x <= INFLUENCE_RADIUS; ++x) {
                if (std::abs(x) + std::abs(y) <= INFLUENCE_RADIUS) {
                    result.emplace_back(x, y);
                }
            }
        }
        return result;
    }

    void computeIndicies()
    {
        if (JD::graphics::particles_loc == nullptr || JD::graphics::offsets == nullptr || JD::graphics::cells_ctr == nullptr) {
            return;
        }
        waitForQueue();
        current_positions.fill(0);
        for (std::size_t index = 0; index < JD::floaters::FLOATER_AMT; ++index) {
            const auto cell = cellIndex(JD::floaters::floatersA.x[index], JD::floaters::floatersA.y[index]);
            if (!cell.has_value()) {
                continue;
            }
            const int position = current_positions[*cell];
            if (position >= JD::graphics::cells_ctr[*cell]) {
                continue;
            }
            JD::graphics::particles_loc[JD::graphics::offsets[*cell] + position] = static_cast<int>(index);
            current_positions[*cell] = position + 1;
        }
    }

    void computeBlockIndicies()
    {
        computeIndicies();
    }

    void rebuild()
    {
        offsetsCreation();
        computeIndicies();
    }
}
