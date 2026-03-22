#include <iostream>

#include "floaters.hpp"
#include "graphics.hpp"
#include "struct.hpp"

namespace JD::floaters
{

    floaters_soa floatersA = {
        new float [ FLOATER_AMT ],
        new float [ FLOATER_AMT ],
        new float [ FLOATER_AMT ],
        new float [ FLOATER_AMT ],
        new float [ FLOATER_AMT ],
        new float [ FLOATER_AMT ],
        new float [ FLOATER_AMT ],
        new float [ FLOATER_AMT ],
        new float [ FLOATER_AMT ],
        new float [ FLOATER_AMT ],
        new float [ FLOATER_AMT ],
        new float [ FLOATER_AMT ],
        new float [ FLOATER_AMT ],
        new bool  [ FLOATER_AMT ]
    };
    block*   blocks    = new block  [ BLOCK_AMT   ];


    void init(float spawn_x, float spawn_y, const std::vector<SpawnBox>& fluidBoxes, const std::vector<SpawnBox>& ghostBoxes)
    {
        initFloaters(spawn_x, spawn_y, fluidBoxes, ghostBoxes);
        initBlockRegions();
    }

    void initFloaters(float spawn_x, float spawn_y, const std::vector<SpawnBox>& fluidBoxes, const std::vector<SpawnBox>& ghostBoxes)
    {
        // initialize EVERYTHING to disabled out-of-bounds first
        for(size_t i = 0; i < FLOATER_AMT; i++)
        {
            floatersA.x[i] = -1000.0f;
            floatersA.y[i] = -1000.0f;
            floatersA.v_x[i] = 0;
            floatersA.v_y[i] = 0;
            floatersA.mass[i] = PARTICLE_MASS;
            floatersA.density[i] = PARTICLE_REFERENCE_DENSITY;
            floatersA.enabled[i] = false;
        }

        size_t current_fluid_i = 0;

        // fluid spawn
        if (!fluidBoxes.empty()) {
            size_t particles_per_box = DESIRED_FLOATERS / fluidBoxes.size();
            for (auto const& box : fluidBoxes) {
                size_t particles_in_this_box = 0;
                for (float y = box.y; y < box.y + box.h; y += 2.1f) {
                    for (float x = box.x; x < box.x + box.w; x += 2.1f) {
                        if (current_fluid_i >= DESIRED_FLOATERS || particles_in_this_box >= particles_per_box) break;
                        floatersA.x[current_fluid_i] = x;
                        floatersA.y[current_fluid_i] = y;
                        floatersA.v_x[current_fluid_i] = 0;
                        floatersA.v_y[current_fluid_i] = 0;
                        floatersA.enabled[current_fluid_i] = true;
                        current_fluid_i++;
                        particles_in_this_box++;
                    }
                    if (current_fluid_i >= DESIRED_FLOATERS || particles_in_this_box >= particles_per_box) break;
                }
            }
        } else if (spawn_x >= 0 && spawn_y >= 0) {
            int side = 90;
            for(size_t i = 0; i < DESIRED_FLOATERS; i++)
            {
                floatersA.x[i] = spawn_x + (i % side) * 2.1f;
                floatersA.y[i] = spawn_y + (i / side) * 2.1f;
                floatersA.v_x[i] = 0;
                floatersA.v_y[i] = 0;
                floatersA.enabled[i] = true;
                current_fluid_i++;
            }
        }

        size_t current_ghost_i = DESIRED_FLOATERS;

        // ghost boxes spawn
        for (auto const& box : ghostBoxes) {
            for (float y = box.y; y < box.y + box.h; y += (float)PARTICLE_GHOST_DENSITY) {
                for (float x = box.x; x < box.x + box.w; x += (float)PARTICLE_GHOST_DENSITY) {
                    if (current_ghost_i >= FLOATER_AMT) break;
                    floatersA.x[current_ghost_i] = x;
                    floatersA.y[current_ghost_i] = y;
                    floatersA.v_x[current_ghost_i] = 0;
                    floatersA.v_y[current_ghost_i] = 0;
                    floatersA.enabled[current_ghost_i] = false;
                    current_ghost_i++;
                }
                if (current_ghost_i >= FLOATER_AMT) break;
            }
            if (current_ghost_i >= FLOATER_AMT) break;
        }

        // shell ghost particles (only if no ghost boxes were provided, or as filler)
        // actually, let's always add shells but start from current_ghost_i
        int shell_idx = 0;
        while (current_ghost_i < FLOATER_AMT) {
            int x0 = BUFFER_PADDING - shell_idx;
            int y0 = BUFFER_PADDING - shell_idx;
            int x1 = BUFFER_PADDING + BUFFER_WORKING + shell_idx;
            int y1 = BUFFER_PADDING + BUFFER_WORKING + shell_idx;

            // std::cout << current_ghost_i << ' ';

            // Top edge
            for (int x = x0; x < x1 && current_ghost_i < FLOATER_AMT; x += PARTICLE_GHOST_DENSITY) {
                floatersA.x[current_ghost_i] = (float)x;
                floatersA.y[current_ghost_i] = (float)y0;
                floatersA.v_x[current_ghost_i] = 0;
                floatersA.v_y[current_ghost_i] = 0;
                floatersA.enabled[current_ghost_i] = false;
                current_ghost_i++;
            }
            // Right edge
            for (int y = y0; y < y1 && current_ghost_i < FLOATER_AMT; y += PARTICLE_GHOST_DENSITY) {
                floatersA.x[current_ghost_i] = (float)x1;
                floatersA.y[current_ghost_i] = (float)y;
                floatersA.v_x[current_ghost_i] = 0;
                floatersA.v_y[current_ghost_i] = 0;
                floatersA.enabled[current_ghost_i] = false;
                current_ghost_i++;
            }
            // Bottom edge
            for (int x = x1; x > x0 && current_ghost_i < FLOATER_AMT; x -= PARTICLE_GHOST_DENSITY) {
                floatersA.x[current_ghost_i] = (float)x;
                floatersA.y[current_ghost_i] = (float)y1;
                floatersA.v_x[current_ghost_i] = 0;
                floatersA.v_y[current_ghost_i] = 0;
                floatersA.enabled[current_ghost_i] = false;
                current_ghost_i++;
            }
            // Left edge
            for (int y = y1; y > y0 && current_ghost_i < FLOATER_AMT; y -= PARTICLE_GHOST_DENSITY) {
                floatersA.x[current_ghost_i] = (float)x0;
                floatersA.y[current_ghost_i] = (float)y;
                floatersA.v_x[current_ghost_i] = 0;
                floatersA.v_y[current_ghost_i] = 0;
                floatersA.enabled[current_ghost_i] = false;
                current_ghost_i++;
            }
            shell_idx += PARTICLE_GHOST_DENSITY;
            if (x0 <= 0 || y0 <= 0) break; 
        }
    
    }


    void drawFloaters()
    {
        for(size_t i = 0; i < FLOATER_AMT; i++)
        {
            int px = (int)floatersA.x[i];
            int py = (int)floatersA.y[i];
            
            if (px >= 0 && px < BUFFER_WIDTH && py >= 0 && py < BUFFER_HEIGHT) {
                int idx = px * BYTES_PER_PIXEL + py * BUFFER_WIDTH * BYTES_PER_PIXEL;
                JD::graphics::static_rgb_buffer[idx+1] = 250;
            }
        }
    }

    // each block stores flat incides of all grid cells within influence_radius
    // in axis directions (2*IR+1)^2
    // must match kernel radii so ghost aprticles placed just outside region
    // are visible to fluid particles at
    // border
    //
    __attribute__ ((noinline))
    void initBlockRegions() {
        const int W     = (int)BUFFER_LINE;
        const int H     = (int)(BLOCK_AMT / W);
        const int IR    = INFLUENCE_RADIUS;
        const int COUNT = BLOCK_NEIGHBOR_COUNT;

        for (size_t i = 0; i < BLOCK_AMT; i++) {
            int row   = (int)(i / W);
            int col   = (int)(i % W);
            int count = 0;

            for (int dy = -IR; dy <= IR; dy++) {
                for (int dx = -IR; dx <= IR; dx++) {
                    int targetRow   = row + dy;
                    int targetCol   = col + dx;
                    int targetIndex = (int)i + dy * W + dx;

                    if (targetRow   >= 0 && targetRow   < H  &&
                        targetCol   >= 0 && targetCol   < W  &&
                        targetIndex >= 0 && targetIndex < (int)BLOCK_AMT) {
                        blocks[i].regions[count++] = (uint32_t)targetIndex;
                    } else {
                        blocks[i].regions[count++] = (uint32_t)INT_MAX;
                    }
                }
            }
            // int_max for everything else
            // this is for people who are bad at coding
            // (I am bad at coding)
            while (count < COUNT) blocks[i].regions[count++] = (uint32_t)INT_MAX;
        }
    }
}
