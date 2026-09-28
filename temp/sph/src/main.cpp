#include "cli.hpp"
#include "floaters.hpp"
#include "graphics.hpp"
#include "gravity.hpp"
#include "logging.hpp"
#include "poly6.hpp"
#include "settings.hpp"
#include "simulate.hpp"
#include "spiky_k.hpp"
#include "spatial.hpp"
#include "sycl.hpp"
#include "viscosity_k.hpp"

#ifdef USE_SDL
#include <SDL2/SDL.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>

namespace
{
    void simulateFloaters()
    {
        JD::simulate::computeDensity<JD::Poly6_k::smoothing>(
            JD::graphics::offsets,
            JD::graphics::cells_ctr,
            JD::graphics::particles_loc,
            JD::floaters::BLOCK_NEIGHBOR_COUNT,
            JD::floaters::blocks,
            JD::floaters::floatersA,
            PARTICLE_SIZE,
            JD::sycl::compute_queue);
        JD::simulate::computePressureForce<JD::Spiky_k::gradient>(
            JD::graphics::offsets,
            JD::graphics::cells_ctr,
            JD::graphics::particles_loc,
            JD::floaters::BLOCK_NEIGHBOR_COUNT,
            JD::floaters::blocks,
            JD::floaters::floatersA,
            PARTICLE_SIZE,
            JD::sycl::compute_queue);
        JD::simulate::computeViscosity<JD::Viscosity_k::laplacian>(
            JD::graphics::offsets,
            JD::graphics::cells_ctr,
            JD::graphics::particles_loc,
            JD::floaters::BLOCK_NEIGHBOR_COUNT,
            JD::floaters::blocks,
            JD::floaters::floatersA,
            PARTICLE_SIZE,
            JD::sycl::compute_queue);
        JD::simulate::applyYAccelerationToAllParticles<JD::gravity::gravityAcceleration>(JD::floaters::floatersA);
        JD::simulate::integrate(JD::floaters::floatersA, PARTICLE_SIZE, JD::sycl::compute_queue);
    }
}

int main(int argc, char** argv)
{
    const JD::cli::Result parsed = JD::cli::parse(argc, argv);
    if (!parsed.ok) {
        std::cerr << parsed.error << "\n" << JD::cli::usage();
        return 2;
    }
    if (parsed.options.help) {
        std::cout << JD::cli::usage();
        return 0;
    }

    bool headless = parsed.options.headless;
#ifndef USE_SDL
    headless = true;
#endif

    std::random_device device;
    std::srand(device());

    try {
        JD::graphics::initGrid();
        JD::graphics::initBuffers();
        JD::floaters::init(-1.0f, -1.0f, parsed.options.fluid_boxes, parsed.options.ghost_boxes);
        JD::spatial::rebuild();
        if (!JD::logging::init()) {
            std::cerr << "could not initialize simulation logging\n";
            JD::floaters::shutdown();
            JD::graphics::shutdown();
            return 1;
        }

#ifdef USE_SDL
        SDL_Window* window = nullptr;
        SDL_Surface* screen_surface = nullptr;
        SDL_Surface* buffer_surface = nullptr;
        SDL_Rect view_rect{0, 0, std::min(WINDOW_WIDTH, BUFFER_WIDTH), std::min(WINDOW_HEIGHT, BUFFER_HEIGHT)};
        if (!headless) {
            if (SDL_Init(SDL_INIT_VIDEO) != 0) {
                std::cerr << "SDL could not initialize: " << SDL_GetError() << '\n';
                JD::logging::finish();
                JD::floaters::shutdown();
                JD::graphics::shutdown();
                return 1;
            }
            window = SDL_CreateWindow("Viewport Render", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
            if (window == nullptr) {
                std::cerr << "SDL could not create a window: " << SDL_GetError() << '\n';
                SDL_Quit();
                JD::logging::finish();
                JD::floaters::shutdown();
                JD::graphics::shutdown();
                return 1;
            }
            buffer_surface = SDL_CreateRGBSurfaceFrom(JD::graphics::static_rgb_buffer, BUFFER_WIDTH, BUFFER_HEIGHT, 24, BUFFER_WIDTH * BYTES_PER_PIXEL, 0x00ff0000u, 0x0000ff00u, 0x000000ffu, 0x00000000u);
            screen_surface = SDL_GetWindowSurface(window);
            if (buffer_surface == nullptr || screen_surface == nullptr) {
                std::cerr << "SDL could not create drawing surfaces: " << SDL_GetError() << '\n';
                if (buffer_surface != nullptr) SDL_FreeSurface(buffer_surface);
                SDL_DestroyWindow(window);
                SDL_Quit();
                JD::logging::finish();
                JD::floaters::shutdown();
                JD::graphics::shutdown();
                return 1;
            }
        }
#endif

        std::cout << "MODEL_VARIANT: " << JD::logging::modelVariant() << '\n';
        std::cout << "BUFFER_WIDTH: " << BUFFER_WIDTH << '\n';
        std::cout << "BUFFER_HEIGHT: " << BUFFER_HEIGHT << '\n';
        std::cout << "HEADLESS: " << (headless ? "ON" : "OFF") << '\n';
        std::cout << "RENDER_MODE: " << (parsed.options.render_mode == JD::cli::RenderMode::Density ? "density" : "particles") << '\n';
        std::cout << "DATA_ROOT: " << JD::logging::sessionDirectory() << '\n';
        std::cout << std::fixed << std::setprecision(3);

        bool quit = false;
        int frame = 0;
        while (!quit && frame < parsed.options.frame_count) {
            const auto frame_start = std::chrono::high_resolution_clock::now();
#ifdef USE_SDL
            if (!headless) {
                SDL_Event event{};
                while (SDL_PollEvent(&event) != 0) {
                    if (event.type == SDL_QUIT) {
                        quit = true;
                    } else if (event.type == SDL_KEYDOWN) {
                        switch (event.key.keysym.sym) {
                        case SDLK_UP: view_rect.y -= 10; break;
                        case SDLK_DOWN: view_rect.y += 10; break;
                        case SDLK_LEFT: view_rect.x -= 10; break;
                        case SDLK_RIGHT: view_rect.x += 10; break;
                        default: break;
                        }
                    }
                }
                view_rect.x = std::max(0, std::min(view_rect.x, BUFFER_WIDTH - view_rect.w));
                view_rect.y = std::max(0, std::min(view_rect.y, BUFFER_HEIGHT - view_rect.h));
                if (quit) {
                    break;
                }
                std::fill(JD::graphics::static_rgb_buffer, JD::graphics::static_rgb_buffer + static_cast<std::size_t>(BUFFER_WIDTH) * BUFFER_HEIGHT * BYTES_PER_PIXEL, static_cast<std::uint8_t>(0));
            }
#endif
            const auto grid_start = std::chrono::high_resolution_clock::now();
            JD::spatial::rebuild();
            const auto grid_end = std::chrono::high_resolution_clock::now();
#ifdef USE_SDL
            if (!headless) {
                if (parsed.options.render_mode == JD::cli::RenderMode::Particles) {
                    JD::floaters::drawFloaters();
                    JD::graphics::computeStrengths();
                    JD::graphics::drawConnections();
                    SDL_BlitSurface(buffer_surface, &view_rect, screen_surface, nullptr);
                    SDL_UpdateWindowSurface(window);
                }
            }
#endif
            const auto simulation_start = std::chrono::high_resolution_clock::now();
            simulateFloaters();
            const auto simulation_end = std::chrono::high_resolution_clock::now();
            JD::spatial::rebuild();
            JD::logging::log(static_cast<std::size_t>(frame + 1),
                             JD::graphics::offsets,
                             JD::graphics::cells_ctr,
                             JD::graphics::particles_loc,
                             JD::floaters::BLOCK_NEIGHBOR_COUNT,
                             JD::floaters::blocks,
                             JD::floaters::floatersA,
                             PARTICLE_SIZE,
                             JD::sycl::compute_queue);
#ifdef USE_SDL
            if (!headless && parsed.options.render_mode == JD::cli::RenderMode::Density) {
                JD::graphics::drawDensity(JD::logging::densityFrame());
                SDL_BlitSurface(buffer_surface, &view_rect, screen_surface, nullptr);
                SDL_UpdateWindowSurface(window);
            }
#endif
            ++frame;
            const auto frame_end = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double, std::milli> grid_ms = grid_end - grid_start;
            const std::chrono::duration<double, std::milli> simulation_ms = simulation_end - simulation_start;
            const std::chrono::duration<double, std::milli> total_ms = frame_end - frame_start;
            if (frame == 1 || frame % 10 == 0 || frame == parsed.options.frame_count) {
                std::cout << "Frame " << (frame - 1)
                          << " | Grid: " << grid_ms.count() << "ms"
                          << " | Sim: " << simulation_ms.count() << "ms"
                          << " | Total: " << total_ms.count() << "ms\n";
            }
        }

#ifdef USE_SDL
        if (!headless) {
            SDL_FreeSurface(buffer_surface);
            SDL_DestroyWindow(window);
            SDL_Quit();
        }
#endif
        JD::logging::finish();
        JD::floaters::shutdown();
        JD::graphics::shutdown();
    } catch (const std::exception& error) {
        std::cerr << "simulation initialization failed: " << error.what() << '\n';
        JD::logging::finish();
        JD::floaters::shutdown();
        JD::graphics::shutdown();
        return 1;
    }
    return 0;
}
