#ifndef _SIMULATE_HPP
#define _SIMULATE_HPP

#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <climits>

#include "struct.hpp"
#include "settings.hpp"
#include "floaters.hpp"
#include "math.hpp"


// diagnostic stuff!!!
struct SimDiag {
    // Fluid particles
    int   fluid_total   = 0;
    int   fluid_in_box  = 0;
    int   fluid_escaped = 0;
    float fluid_max_v   = 0.f;
    float fluid_max_d   = 0.f;
    float fluid_max_p   = 0.f;
    float fluid_avg_d   = 0.f;
    float fluid_avg_p   = 0.f;

    int   ghost_total        = 0;
    int   ghost_in_grid      = 0;
    int   ghost_max_pressure = 0;
    float ghost_avg_pressure = 0.f;

    int near_left   = 0;
    int near_right  = 0;
    int near_top    = 0;
    int near_bottom = 0;
};

template <typename Particles>
inline SimDiag collectDiagnostics(const Particles& p, int frame)
{
    SimDiag d;
    const float lo_x = (float)BUFFER_PADDING;
    const float hi_x = (float)(BUFFER_PADDING + BUFFER_WORKING);
    const float lo_y = (float)BUFFER_PADDING;
    const float hi_y = (float)(BUFFER_PADDING + BUFFER_WORKING);
    const float wall_thresh = PARTICLE_SIZE;

    float sum_d_fluid = 0.f, sum_p_fluid = 0.f;
    float sum_p_ghost = 0.f;

    for (size_t i = 0; i < FLOATER_AMT; i++) {
        if (p.enabled[i]) {
            d.fluid_total++;
            float vabs = std::abs(p.v_x[i]) + std::abs(p.v_y[i]);
            d.fluid_max_v = std::max(d.fluid_max_v, vabs);
            d.fluid_max_d = std::max(d.fluid_max_d, p.density[i]);
            d.fluid_max_p = std::max(d.fluid_max_p, p.pressure[i]);
            sum_d_fluid  += p.density[i];
            sum_p_fluid  += p.pressure[i];

            bool in_box = (p.x[i] >= lo_x && p.x[i] < hi_x &&
                           p.y[i] >= lo_y && p.y[i] < hi_y);
            if (in_box)  d.fluid_in_box++;
            else         d.fluid_escaped++;

            if (p.x[i] - lo_x < wall_thresh) d.near_left++;
            if (hi_x - p.x[i] < wall_thresh) d.near_right++;
            if (p.y[i] - lo_y < wall_thresh) d.near_top++;
            if (hi_y - p.y[i] < wall_thresh) d.near_bottom++;
        } else {
            d.ghost_total++;
            bool in_grid = (p.x[i] >= 0 && p.x[i] < (float)BUFFER_WIDTH &&
                            p.y[i] >= 0 && p.y[i] < (float)BUFFER_HEIGHT);
            if (in_grid) d.ghost_in_grid++;
            if (p.pressure[i] > (float)d.ghost_max_pressure)
                d.ghost_max_pressure = (int)p.pressure[i];
            sum_p_ghost += p.pressure[i];
        }
    }

    if (d.fluid_total  > 0) { d.fluid_avg_d = sum_d_fluid / d.fluid_total;
                               d.fluid_avg_p = sum_p_fluid / d.fluid_total; }
    if (d.ghost_total  > 0)   d.ghost_avg_pressure = sum_p_ghost / d.ghost_total;
    return d;
}

inline void printDiagnostics(const SimDiag& d, int frame)
{
    // if there is an error regarding this, rewrite it, there this was written using ai lol
    //
    std::cerr << std::fixed << std::setprecision(3);
    std::cerr << "=== FRAME " << frame << " ===\n";
    std::cerr << "  FLUID  total=" << d.fluid_total
              << "  in-box=" << d.fluid_in_box
              << "  escaped=" << d.fluid_escaped << "\n";
    std::cerr << "         max_v=" << d.fluid_max_v
              << "  max_d=" << d.fluid_max_d
              << "  max_p=" << d.fluid_max_p
              << "  avg_d=" << d.fluid_avg_d
              << "  avg_p=" << d.fluid_avg_p << "\n";
    std::cerr << "  WALLS  near_left=" << d.near_left
              << "  near_right=" << d.near_right
              << "  near_top=" << d.near_top
              << "  near_bottom=" << d.near_bottom << "\n";
    std::cerr << "  GHOST  total=" << d.ghost_total
              << "  in-grid=" << d.ghost_in_grid
              << "  max_p=" << d.ghost_max_pressure
              << "  avg_p=" << d.ghost_avg_pressure << "\n";
    // Warn if ghosts are not registering in the grid at all
    if (d.ghost_total > 0 && d.ghost_in_grid == 0) {
        std::cerr << "  [WARN] NO ghost particles are in the spatial grid! "
                     "Boundary will not work.\n";
    }
    // Warn if large number of fluid escaped
    if (d.fluid_escaped > d.fluid_total / 10) {
        std::cerr << "  [WARN] " << d.fluid_escaped
                  << " fluid particles are outside the working region!\n";
    }
}


namespace JD::simulate
{
    struct SpatialView {
        int* offsets;
        int* counts;
        int* locs;
        JD::floaters::block* blocks;
        int  region_amt;
        float grid_spacing;
        int   grid_line;

        template<typename F>
        inline void for_each_neighbor(float x, float y, F&& func) const {
            int bx = (int)(x / grid_spacing);
            int by = (int)(y / grid_spacing);
            
            if (bx >= 0 && bx < grid_line && by >= 0 && by < grid_line) {
                size_t idx = (size_t)(bx + by * grid_line);

                for (int j = 0; j < region_amt; j++) {
                    int idx_r = (int)blocks[idx].regions[j];
                    if (idx_r == INT_MAX) continue;

                    int idx_o = offsets[idx_r];
                    for (int k = 0; k < counts[idx_r]; k++) {
                        func(locs[idx_o + k]);
                    }
                }
            }
        }
    };

    template <auto KernelFunc, typename Particles>
    void computeDensity(const SpatialView& grid,
                        Particles& p,
                        float h)
    {
#pragma omp parallel for num_threads(16)
        for (size_t i = 0; i < FLOATER_AMT; i++)
        {
            float temp = 0.0f;
            float x    = p.x[i];
            float y    = p.y[i];

            grid.for_each_neighbor(x, y, [&](int j) {
                float dx = p.x[j] - x;
                float dy = p.y[j] - y;
                float dist_sq = dx*dx + dy*dy;
                temp += p.mass[j] * KernelFunc(dist_sq, h);
            });

            // ghost are on wall, their densities are a bit higher to 
            // ya
            if (!p.enabled[i]) temp *= 8.0f;

            p.density[i]  = temp;
            p.pressure[i] = std::max(0.0f,
                PARTICLE_BULK_MODULUS * (p.density[i] - PARTICLE_REFERENCE_DENSITY));
        }
    }


    template <auto KernelGrad, typename Particles>
    void computePressureForce(const SpatialView& grid,
                              Particles& p,
                              float h)
    {
#pragma omp parallel for num_threads(16)
        for (size_t i = 0; i < FLOATER_AMT; i++) {
            if (!p.enabled[i]) continue; // Ghosts don't receive forces

            float x  = p.x[i];
            float y  = p.y[i];
            
            grid.for_each_neighbor(x, y, [&](int j) {
                if (i == (size_t)j) return;

                float dx    = x - p.x[j];
                float dy    = y - p.y[j];
                float r_sq  = dx*dx + dy*dy;

                float dist = std::sqrt(r_sq);
                float r_norm = dist/h;

                if (r_sq > 0 && r_sq < h*h) {
                    if (!p.enabled[j])
                    {
                        if(r_norm < PARTICLE_SIZE)
                        {
                            float k_repulsion = 30.5 * PARTICLE_BULK_MODULUS;
                            float force_mag = k_repulsion * (1.0f - r_norm) / (r_sq + 0.01f);

                            p.a_x[i] += force_mag * (dx / dist);
                            p.a_y[i] += force_mag * (dy / dist);
                        
                            float friction_coeff = 0.1f;
                            float dvx = p.v_x[i] - p.v_x[j]; 
                            float dvy = p.v_y[i] - p.v_y[j];

                            p.a_x[i] -= friction_coeff * dvx;
                            p.a_y[i] -= friction_coeff * dvy;
                        }
                    }
                    else {
                        force grad_f;
                        KernelGrad(dx, dy, r_sq, h, grad_f);

                        float rho_i = JD::math::ffast_max(p.density[i], 1e-6f);
                        float rho_j = JD::math::ffast_max(p.density[j], 1e-6f);
                        
                        float p_i = p.pressure[i];
                        float p_j = (!p.enabled[j]) ? p_i * 1.0f : p.pressure[j];

                        float p_term = (p_i + p_j) / (rho_i * rho_j);

                        p.a_x[i] -= p.mass[j] * p_term * grad_f.x;
                        p.a_y[i] -= p.mass[j] * p_term * grad_f.y;
                    }
                }
            });
        }
    }

    template <auto KernelLap, typename Particles>
    void computeViscosity(const SpatialView& grid,
                          Particles& p,
                          float h) {
#pragma omp parallel for num_threads(16)
        for (size_t i = 0; i < FLOATER_AMT; i++) {
            if (!p.enabled[i]) continue;

            float x  = p.x[i];
            float y  = p.y[i];
            
            grid.for_each_neighbor(x, y, [&](int j) {
                if (i == (size_t)j) return;

                float dx   = x - p.x[j];
                float dy   = y - p.y[j];
                float r_sq = dx*dx + dy*dy;
                if (r_sq > 0 && r_sq < h*h) {
                    float lap   = KernelLap(r_sq, h);
                    float rho_j = std::max(p.density[j], 1e-6f);
                    float v_mod = PARTICLE_VISCOSITY * (p.mass[j] / rho_j);
                    p.a_x[i] += v_mod * (p.v_x[j] - p.v_x[i]) * lap;
                    p.a_y[i] += v_mod * (p.v_y[j] - p.v_y[i]) * lap;
                }
            });
        }
    }

    // apply accel to activ
    template <auto function, typename Particles>
    void applyYAccelerationToAllParticles(Particles& p)
    {
        float valueToApply = function();
        for (size_t i = 0; i < FLOATER_AMT; i++)
        {
            if (!p.enabled[i]) break;
            p.a_y[i] += valueToApply;
        }
    }
    

    template <typename Particles>
    void integrate(Particles& p) 
    {
#pragma omp parallel for num_threads(16)
        for (size_t i = 0; i < FLOATER_AMT; i++) 
        {
            if (!p.enabled[i]) continue; // ghosts stay fixed

            p.v_x[i] += p.a_x[i] * PARTICLE_TIME_STEP; 
            p.v_y[i] += p.a_y[i] * PARTICLE_TIME_STEP;

            // Velocity clamp – prevents tunnelling through boundaries
            if (p.v_x[i] >  PARTICLE_MAX_V) p.v_x[i] =  PARTICLE_MAX_V;
            if (p.v_x[i] < -PARTICLE_MAX_V) p.v_x[i] = -PARTICLE_MAX_V;
            if (p.v_y[i] >  PARTICLE_MAX_V) p.v_y[i] =  PARTICLE_MAX_V;
            if (p.v_y[i] < -PARTICLE_MAX_V) p.v_y[i] = -PARTICLE_MAX_V;

            p.x[i] += p.v_x[i] * PARTICLE_TIME_STEP; 
            p.y[i] += p.v_y[i] * PARTICLE_TIME_STEP;
            p.a_x[i] = 0; 
            p.a_y[i] = 0;
        }
    }
}

#endif
