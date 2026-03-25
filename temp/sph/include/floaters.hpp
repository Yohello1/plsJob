#ifndef _FLOATERS_HPP
#define _FLOATERS_HPP

#include <climits>

#include "ghost.hpp"
#include "struct.hpp"
#include "settings.hpp"

#include <vector>

namespace JD::floaters
{
    static constexpr inline size_t BLOCK_AMT = BUFFER_LINE*BUFFER_LINE;
    static constexpr inline int BLOCK_NEIGHBOR_DIM   = 2 * INFLUENCE_RADIUS + 1;
    static constexpr inline int BLOCK_NEIGHBOR_COUNT = BLOCK_NEIGHBOR_DIM * BLOCK_NEIGHBOR_DIM;
   
    struct block
    {
        uint32_t regions[BLOCK_NEIGHBOR_COUNT];
    };

    extern floaters_soa floatersA;
    extern block*   blocks; 

    void init(float spawn_x = -1.0f, float spawn_y = -1.0f, 
              const std::vector<SpawnBox>& fluidBoxes = {}, 
              const std::vector<SpawnBox>& ghostBoxes = {});

    void initFloaters(float spawn_x = -1.0f, float spawn_y = -1.0f,
                      const std::vector<SpawnBox>& fluidBoxes = {},
                      const std::vector<SpawnBox>& ghostBoxes = {});
    void drawFloaters();

    void initBlockRegions();

}


#endif // _FLOATERS_HPP
