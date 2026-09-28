#ifndef JD_FLOATERS_HPP
#define JD_FLOATERS_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ghost.hpp"
#include "settings.hpp"
#include "struct.hpp"

namespace JD::floaters
{
    inline constexpr std::size_t DESIRED_FLOATERS = ::DESIRED_FLOATERS;
    inline constexpr std::size_t GHOST_FLOATERS = ::GHOST_FLOATERS;
    inline constexpr std::size_t FLOATER_AMT = ::FLOATER_AMT;
    inline constexpr int FLOATER_SPEED = ::FLOATER_SPEED;
    inline constexpr std::size_t BLOCK_AMT = static_cast<std::size_t>(BUFFER_LINE) * BUFFER_LINE;
    inline constexpr int BLOCK_NEIGHBOR_DIM = 2 * INFLUENCE_RADIUS + 1;
    inline constexpr int BLOCK_NEIGHBOR_COUNT = BLOCK_NEIGHBOR_DIM * BLOCK_NEIGHBOR_DIM;

    struct block
    {
        std::uint32_t regions[BLOCK_NEIGHBOR_COUNT];
    };

    extern floaters_soa floatersA;
    extern block* blocks;

    void init(float spawn_x = -1.0f,
              float spawn_y = -1.0f,
              const std::vector<SpawnBox>& fluid_boxes = {},
              const std::vector<SpawnBox>& ghost_boxes = {});
    void initFloaters(float spawn_x = -1.0f,
                      float spawn_y = -1.0f,
                      const std::vector<SpawnBox>& fluid_boxes = {},
                      const std::vector<SpawnBox>& ghost_boxes = {});
    void drawFloaters();
    void initBlockRegions();
    void shutdown();
}

#endif
