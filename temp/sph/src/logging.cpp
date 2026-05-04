#include "logging.hpp"
#include "settings.hpp"
#include "floaters.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <random>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <sycl/sycl.hpp>

namespace JD::logging
{
    static std::ofstream _log_file;
    static std::string _logging_dir;

    static std::string _get_log_dirname() {
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm bt = *std::localtime(&now_time);

        std::ostringstream oss;
        oss << std::put_time(&bt, "%Y%m%d_%H%M%S");

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 0xFFFF);
        
        oss << "_" << std::hex << std::setw(4) << std::setfill('0') << dis(gen);

        if (const char* array_id = std::getenv("SLURM_ARRAY_TASK_ID")) {
            oss << "_task" << array_id;
        }

        return oss.str();
    }

    void init()
    {
        const char* data_root = std::getenv("SPH_DATA_ROOT");
        std::string base = (data_root && strlen(data_root) > 0) ? std::string(data_root) : "data";
        
        _logging_dir = base + "/" + _get_log_dirname();
        if (!std::filesystem::exists(_logging_dir)) 
        {
            std::filesystem::create_directories(_logging_dir);
        }

        _log_file.open(_logging_dir + "/sim_data.bin", std::ios::binary);
    }
   
    // aint no way Im passing in the same data which I do for the 
    // simulation functions...
    // FOR A LOGGING FUNCTION
    void log(size_t i,
             int* offsets_in,
             int* cells_ctr_in,
             int* particles_loc_in, 
             int region_amt, 
             JD::floaters::block* blocks_in,
             floaters_soa particles_in,
             float h_in, 
             ::sycl::queue& q)
    {
        if (!_log_file.is_open()) return;

        // Grid size NxN
        const int N_W = BUFFER_WIDTH;
        const int N_H = BUFFER_HEIGHT;

        float* sum_d = sycl::malloc_shared<float>(N_W * N_H, q);
        q.fill(sum_d, 0.0f, N_W * N_H).wait();
        std::vector<float> sum_vx(N_W * N_H, 0.0f);
        std::vector<float> sum_vy(N_W * N_H, 0.0f);
        std::vector<int> count(N_W * N_H, 0);
        std::vector<int> mask(N_W * N_H, 0);

        auto& f = JD::floaters::floatersA;

        for (size_t p_idx = 0; p_idx < FLOATER_AMT; ++p_idx) {
            int px = static_cast<int>(f.x[p_idx]);
            int py = static_cast<int>(f.y[p_idx]);

            if (px >= 0 && px < N_W && py >= 0 && py < N_H) {
                int idx = py * N_W + px;
                if (f.enabled[p_idx]) {
                    // sum_d[idx] += f.density[p_idx];
                    sum_vx[idx] += f.v_x[p_idx];
                    sum_vy[idx] += f.v_y[p_idx];
                    count[idx]++;
                }
                
                if (p_idx >= DESIRED_FLOATERS) {
                    mask[idx] = 1;
                }
            }
        }

        // In going to cry when writting the density summation part lol
        // as much as I want to just use the m^2*n algo to solve this (triple for loops)
        // that shit gonna take 8,000,000,000 operations...
        // so optimised path we go!

        // we can stick it onto the gpu (sycl) later lol
        // linear hat function = -|x| + 1 {-1 <= x <= 1, else 0}

        // double for loop for all parts, convert to block index and sample particles around 
        // area... how did I do the thingy again, if I reverse order I can save time
        // anyways

            q.parallel_for(sycl::range<2>(BUFFER_HEIGHT, BUFFER_WIDTH), [=](sycl::id<2> idx_id)
            {
                int i = idx_id[1];
                int j = idx_id[0];

                int bx = (int) (i / DISTANCE_BETWEEN_POINTS);
                int by = (int) (j / DISTANCE_BETWEEN_POINTS);
                int idx_d = j * BUFFER_WIDTH + i;

                size_t idx = (size_t)(bx + by * BUFFER_LINE);

                for (int r = 0; r < region_amt; r++) 
                {
                    int idx_r = (int)blocks_in[idx].regions[r];
                    if (idx_r == INT_MAX) continue;

                    int idx_o = offsets_in[idx_r];
                    for (int k = 0; k < cells_ctr_in[idx_r]; k++) 
                    {
                        int floater_idx = particles_loc_in[idx_o + k];
                        float dx = particles_in.x[floater_idx] - i;
                        float dy = particles_in.y[floater_idx] - j;
                        float dist = sycl::fabs(sycl::sqrt(dx*dx + dy*dy));
                
                        float kernel_func = (dist <= 1.0f) ? (1.0f - dist) : 0.0f;

                        sum_d[idx_d] += kernel_func;
                    }
                }
            }).wait();

        auto write_chunk = [&](auto get_val) {
            for (int j = 0; j < N_H * N_W; ++j) {
                float val = static_cast<float>(get_val(j));
                _log_file.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        };

        // Order: Density, V_x, V_y, Mask
        write_chunk([&](int idx) { return sum_d[idx]; });
        write_chunk([&](int idx) { return count[idx] > 0 ? sum_vx[idx] / count[idx] : 0.0f; });
        write_chunk([&](int idx) { return count[idx] > 0 ? sum_vy[idx] / count[idx] : 0.0f; });
        write_chunk([&](int idx) { return static_cast<float>(mask[idx]); });

        _log_file.flush();
        sycl::free(sum_d, q);
    }

    void finish()
    {
        if (_log_file.is_open())
        {
            _log_file.close();
        }
    }
}
