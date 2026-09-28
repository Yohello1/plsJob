#ifndef JD_SIMULATE_HPP
#define JD_SIMULATE_HPP

#include <sycl/sycl.hpp>

#include "floaters.hpp"
#include "settings.hpp"
#include "struct.hpp"

namespace JD::simulate
{
    template <auto KernelFunction>
    void computeDensity(int* offsets_in,
                        int* cells_ctr_in,
                        int* particles_loc_in,
                        int region_amount,
                        JD::floaters::block* blocks_in,
                        floaters_soa particles,
                        float particle_size,
                        ::sycl::queue& queue)
    {
        queue.parallel_for(::sycl::range<1>(JD::floaters::FLOATER_AMT), [=](::sycl::id<1> id) {
            const std::size_t index = static_cast<std::size_t>(id[0]);
            const float x = particles.x[index];
            const float y = particles.y[index];
            const int bx = static_cast<int>(x / DISTANCE_BETWEEN_POINTS);
            const int by = static_cast<int>(y / DISTANCE_BETWEEN_POINTS);
            float density = 0.0f;
            if (bx >= 0 && bx < BUFFER_LINE && by >= 0 && by < BUFFER_LINE) {
                const std::size_t block_index = static_cast<std::size_t>(bx + by * BUFFER_LINE);
                for (int region = 0; region < region_amount; ++region) {
                    const std::uint32_t neighbor = blocks_in[block_index].regions[region];
                    if (neighbor == 0xffffffffu) {
                        continue;
                    }
                    const int neighbor_index = static_cast<int>(neighbor);
                    const int start = offsets_in[neighbor_index];
                    const int count = cells_ctr_in[neighbor_index];
                    for (int item = 0; item < count; ++item) {
                        const int particle = particles_loc_in[start + item];
                        const float dx = particles.x[particle] - x;
                        const float dy = particles.y[particle] - y;
                        const float distance_squared = dx * dx + dy * dy;
                        density += particles.mass[particle] * KernelFunction(distance_squared, particle_size);
                    }
                }
            }
            particles.density[index] = density;
            particles.pressure[index] = ::sycl::fmax(0.0f, PARTICLE_BULK_MODULUS * (density - PARTICLE_REFERENCE_DENSITY));
        }).wait();
    }

    template <auto KernelFunction>
    void computePressureForce(int* offsets_in,
                              int* cells_ctr_in,
                              int* particles_loc_in,
                              int region_amount,
                              JD::floaters::block* blocks_in,
                              floaters_soa particles,
                              float particle_size,
                              ::sycl::queue& queue)
    {
        queue.parallel_for(::sycl::range<1>(JD::floaters::FLOATER_AMT), [=](::sycl::id<1> id) {
            const std::size_t index = static_cast<std::size_t>(id[0]);
            if (!particles.enabled[index]) {
                return;
            }
            const float x = particles.x[index];
            const float y = particles.y[index];
            const int bx = static_cast<int>(x / DISTANCE_BETWEEN_POINTS);
            const int by = static_cast<int>(y / DISTANCE_BETWEEN_POINTS);
            if (bx < 0 || bx >= BUFFER_LINE || by < 0 || by >= BUFFER_LINE) {
                return;
            }
            const std::size_t block_index = static_cast<std::size_t>(bx + by * BUFFER_LINE);
            for (int region = 0; region < region_amount; ++region) {
                const std::uint32_t neighbor = blocks_in[block_index].regions[region];
                if (neighbor == 0xffffffffu) {
                    continue;
                }
                const int neighbor_index = static_cast<int>(neighbor);
                const int start = offsets_in[neighbor_index];
                const int count = cells_ctr_in[neighbor_index];
                for (int item = 0; item < count; ++item) {
                    const int particle = particles_loc_in[start + item];
                    if (particle == static_cast<int>(index)) {
                        continue;
                    }
                    const float dx = x - particles.x[particle];
                    const float dy = y - particles.y[particle];
                    const float distance_squared = dx * dx + dy * dy;
                    if (distance_squared <= 0.0f || distance_squared >= particle_size * particle_size) {
                        continue;
                    }
                    if (!particles.enabled[particle]) {
                        const float distance = ::sycl::sqrt(distance_squared);
                        const float normalized_distance = distance / particle_size;
                        if (normalized_distance < PARTICLE_SIZE) {
                            const float force_magnitude = PARTICLE_REPULSION * (1.0f - normalized_distance) / (distance_squared + 0.01f);
                            particles.a_x[index] += force_magnitude * dx / distance;
                            particles.a_y[index] += force_magnitude * dy / distance;
                            const float friction = 0.1f;
                            particles.a_x[index] -= friction * (particles.v_x[index] - particles.v_x[particle]);
                            particles.a_y[index] -= friction * (particles.v_y[index] - particles.v_y[particle]);
                        }
                    } else {
                        force gradient;
                        KernelFunction(dx, dy, distance_squared, particle_size, gradient);
                        const float rho_i = ::sycl::fmax(particles.density[index], 1.0e-6f);
                        const float rho_j = ::sycl::fmax(particles.density[particle], 1.0e-6f);
                        const float pressure_term = (particles.pressure[index] + particles.pressure[particle]) / (rho_i * rho_j);
                        particles.a_x[index] -= particles.mass[particle] * pressure_term * gradient.x;
                        particles.a_y[index] -= particles.mass[particle] * pressure_term * gradient.y;
                    }
                }
            }
        }).wait();
    }

    template <auto KernelFunction>
    void computeViscosity(int* offsets_in,
                          int* cells_ctr_in,
                          int* particles_loc_in,
                          int region_amount,
                          JD::floaters::block* blocks_in,
                          floaters_soa particles,
                          float particle_size,
                          ::sycl::queue& queue)
    {
        queue.parallel_for(::sycl::range<1>(JD::floaters::FLOATER_AMT), [=](::sycl::id<1> id) {
            const std::size_t index = static_cast<std::size_t>(id[0]);
            if (!particles.enabled[index]) {
                return;
            }
            const float x = particles.x[index];
            const float y = particles.y[index];
            const int bx = static_cast<int>(x / DISTANCE_BETWEEN_POINTS);
            const int by = static_cast<int>(y / DISTANCE_BETWEEN_POINTS);
            if (bx < 0 || bx >= BUFFER_LINE || by < 0 || by >= BUFFER_LINE) {
                return;
            }
            const std::size_t block_index = static_cast<std::size_t>(bx + by * BUFFER_LINE);
            for (int region = 0; region < region_amount; ++region) {
                const std::uint32_t neighbor = blocks_in[block_index].regions[region];
                if (neighbor == 0xffffffffu) {
                    continue;
                }
                const int neighbor_index = static_cast<int>(neighbor);
                const int start = offsets_in[neighbor_index];
                const int count = cells_ctr_in[neighbor_index];
                for (int item = 0; item < count; ++item) {
                    const int particle = particles_loc_in[start + item];
                    if (particle == static_cast<int>(index)) {
                        continue;
                    }
                    const float dx = x - particles.x[particle];
                    const float dy = y - particles.y[particle];
                    const float distance_squared = dx * dx + dy * dy;
                    if (distance_squared <= 0.0f || distance_squared >= particle_size * particle_size) {
                        continue;
                    }
                    const float laplacian = KernelFunction(distance_squared, particle_size);
                    const float rho = ::sycl::fmax(particles.density[particle], 1.0e-6f);
                    const float scale = PARTICLE_VISCOSITY * particles.mass[particle] / rho;
                    particles.a_x[index] += scale * (particles.v_x[particle] - particles.v_x[index]) * laplacian;
                    particles.a_y[index] += scale * (particles.v_y[particle] - particles.v_y[index]) * laplacian;
                }
            }
        }).wait();
    }

    template <auto KernelFunction>
    void computePressureForce(int* offsets_in,
                              int* cells_ctr_in,
                              int* particles_loc_in,
                              JD::floaters::block* blocks_in,
                              floaters_soa particles,
                              float particle_size,
                              ::sycl::queue& queue)
    {
        computePressureForce<KernelFunction>(offsets_in, cells_ctr_in, particles_loc_in, JD::floaters::BLOCK_NEIGHBOR_COUNT, blocks_in, particles, particle_size, queue);
    }

    template <auto KernelFunction>
    void computeViscosity(int* offsets_in,
                          int* cells_ctr_in,
                          int* particles_loc_in,
                          JD::floaters::block* blocks_in,
                          floaters_soa particles,
                          float particle_size,
                          ::sycl::queue& queue)
    {
        computeViscosity<KernelFunction>(offsets_in, cells_ctr_in, particles_loc_in, JD::floaters::BLOCK_NEIGHBOR_COUNT, blocks_in, particles, particle_size, queue);
    }

    template <auto Function>
    void applyYAccelerationToAllParticles(floaters_soa particles)
    {
        const float acceleration = Function();
        for (std::size_t index = 0; index < JD::floaters::FLOATER_AMT; ++index) {
            if (particles.enabled[index]) {
                particles.a_y[index] += acceleration;
            }
        }
    }

    inline void integrate(floaters_soa particles, float particle_size, ::sycl::queue& queue)
    {
        queue.parallel_for(::sycl::range<1>(JD::floaters::FLOATER_AMT), [=](::sycl::id<1> id) {
            const std::size_t index = static_cast<std::size_t>(id[0]);
            if (!particles.enabled[index]) {
                return;
            }
            particles.v_x[index] += particles.a_x[index] * PARTICLE_TIME_STEP;
            particles.v_y[index] += particles.a_y[index] * PARTICLE_TIME_STEP;
            particles.v_x[index] = ::sycl::fmin(PARTICLE_MAX_V, ::sycl::fmax(-PARTICLE_MAX_V, particles.v_x[index]));
            particles.v_y[index] = ::sycl::fmin(PARTICLE_MAX_V, ::sycl::fmax(-PARTICLE_MAX_V, particles.v_y[index]));
            particles.x[index] += particles.v_x[index] * PARTICLE_TIME_STEP;
            particles.y[index] += particles.v_y[index] * PARTICLE_TIME_STEP;
            particles.a_x[index] = 0.0f;
            particles.a_y[index] = 0.0f;
        }).wait();
        (void)particle_size;
    }

    inline void integrate(int* offsets_in,
                          int* cells_ctr_in,
                          int* particles_loc_in,
                          floaters_soa particles,
                          ::sycl::queue& queue)
    {
        (void)offsets_in;
        (void)cells_ctr_in;
        (void)particles_loc_in;
        integrate(particles, PARTICLE_SIZE, queue);
    }
}

#endif
