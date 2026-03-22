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

namespace JD::logging
{
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
        return oss.str();
    }

    void init()
    {
        _logging_dir = _get_log_dirname();
        if (!std::filesystem::exists(_logging_dir)) 
        {
            std::filesystem::create_directories(_logging_dir);
        }
    }
    
    void log(size_t i)
    {
        // Grid size NxN
        const int N_W = BUFFER_WIDTH;
        const int N_H = BUFFER_HEIGHT;

        std::vector<float> sum_d(N_W * N_H, 0.0f);
        std::vector<float> sum_vx(N_W * N_H, 0.0f);
        std::vector<float> sum_vy(N_W * N_H, 0.0f);
        std::vector<int> count(N_W * N_H, 0);

        auto& f = JD::floaters::floatersA;

        for (size_t p_idx = 0; p_idx < FLOATER_AMT; ++p_idx) {
            if (!f.enabled[p_idx]) continue;

            int px = static_cast<int>(f.x[p_idx]);
            int py = static_cast<int>(f.y[p_idx]);

            if (px >= 0 && px < N_W && py >= 0 && py < N_H) {
                int idx = py * N_W + px;
                sum_d[idx] += f.density[p_idx];
                sum_vx[idx] += f.v_x[p_idx];
                sum_vy[idx] += f.v_y[p_idx];
                count[idx]++;
            }
        }

        auto write_csv = [&](const std::string& suffix, auto get_val) {
            std::string path = _logging_dir + "/" + std::to_string(i) + "_" + suffix + ".csv";
            std::ofstream ofs(path);
            if (!ofs.is_open()) return;

            ofs << std::fixed << std::setprecision(3);
            for (int r = 0; r < N_H; ++r) {
                for (int c = 0; c < N_W; ++c) {
                    ofs << get_val(r * N_W + c) << (c == N_W - 1 ? "" : ",");
                }
                ofs << "\n";
            }
        };

        // Density
        write_csv("d", [&](int idx) {
            return count[idx] > 0 ? sum_d[idx] / count[idx] : 0.0f;
        });

        // V_x
        write_csv("v_x", [&](int idx) {
            return count[idx] > 0 ? sum_vx[idx] / count[idx] : 0.0f;
        });

        // V_y
        write_csv("v_y", [&](int idx) {
            return count[idx] > 0 ? sum_vy[idx] / count[idx] : 0.0f;
        });

        // Mask
        write_csv("m", [&](int idx) {
            return count[idx] > 0 ? 1 : 0;
        });
    }
}

