#include "settings.hpp"
#include "struct.hpp"
#include "floaters.hpp"
#include "graphics.hpp"
#include "spatial.hpp"
#include "simulate.hpp"
#include "poly6.hpp"
#include "viscosity_k.hpp"
#include "gravity.hpp"
#include "math.hpp"
#include "spiky_k.hpp"
#include "logging.hpp"
#include "sycl.hpp" 
#ifdef USE_SDL
#include <SDL2/SDL.h>
#endif
#include <omp.h>

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <random>
#include <filesystem>
#include <chrono>
#include <chrono>


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
        JD::floaters::blocks,
        JD::floaters::floatersA,
        PARTICLE_SIZE,
        JD::sycl::compute_queue);
    JD::simulate::computeViscosity<JD::Viscosity_k::laplacian>(
        JD::graphics::offsets,
        JD::graphics::cells_ctr,
        JD::graphics::particles_loc,
        JD::floaters::blocks,
        JD::floaters::floatersA,
        PARTICLE_SIZE,
        JD::sycl::compute_queue);
    JD::simulate::applyYAccelerationToAllParticles<JD::gravity::gravityAcceleration>(
        JD::floaters::floatersA);
    JD::simulate::integrate(
        JD::graphics::offsets,
        JD::graphics::cells_ctr,
        JD::graphics::particles_loc,
        JD::floaters::floatersA,
        JD::sycl::compute_queue);
}


int main(int argc, char** argv) {
    int max_frames = -1;
    std::vector<SpawnBox> fluidBoxes;
    std::vector<SpawnBox> ghostBoxes;

    if (argc > 1) {
        // is first args frame or nah-
        if (argv[1][0] != '-') {
            max_frames = std::atoi(argv[1]);
        }
    }

    bool headless = false;
#ifndef USE_SDL
    headless = true;
#endif

    // parsing!
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--headless") {
            headless = true;
        } else if ((arg == "--fluid" || arg == "-f") && i + 4 < argc) {
            float x = (float)std::atof(argv[++i]);
            float y = (float)std::atof(argv[++i]);
            float w = (float)std::atof(argv[++i]);
            float h = (float)std::atof(argv[++i]);
            fluidBoxes.push_back({x, y, w, h});
        } else if ((arg == "--ghost" || arg == "-g") && i + 4 < argc) {
            float x = (float)std::atof(argv[++i]);
            float y = (float)std::atof(argv[++i]);
            float w = (float)std::atof(argv[++i]);
            float h = (float)std::atof(argv[++i]);
            ghostBoxes.push_back({x, y, w, h});
        }
    }

    JD::graphics::initGrid();
    JD::floaters::init(-1.0f, -1.0f, fluidBoxes, ghostBoxes);
    
    // Find where to put the data
    const char* data_root = std::getenv("SPH_DATA_ROOT");
    std::string base = (data_root && strlen(data_root) > 0) ? std::string(data_root) : "data";
    std::filesystem::create_directories(base + "/frames");
    
    JD::logging::init();

    
    // Robust seeding
    std::random_device rd;
    srand(rd());

    std::cout << "BUFFER_LINE: " << BUFFER_LINE << std::endl;
    std::cout << "DISTANCE_BETWEEN_POINTS: " << DISTANCE_BETWEEN_POINTS << std::endl;
    std::cout << "SIZE_MULTIPLIER: " << SIZE_MULTIPLIER << std::endl;
    std::cout << "PADDING: " << PADDING << std::endl;
    std::cout << "BUFFER_WIDTH:  " << BUFFER_WIDTH << std::endl;
    std::cout << "BUFFER_HEIGHT: " << BUFFER_HEIGHT << std::endl;
    std::cout << "HEADLESS: " << (headless ? "ON" : "OFF") << std::endl;

    std::cout << std::fixed << std::setprecision(2);

    JD::graphics::initBuffers();
#ifdef USE_SDL
    SDL_Window* window = nullptr;
    SDL_Surface* screenSurface = nullptr;
    SDL_Surface* bufferSurface = nullptr;
    SDL_Rect viewRect;

    if (!headless) {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
            return 1;
        }

        window = SDL_CreateWindow(
            "Viewport Render",
            SDL_WINDOWPOS_UNDEFINED,
            SDL_WINDOWPOS_UNDEFINED,
            WINDOW_WIDTH, 
            WINDOW_HEIGHT,
            SDL_WINDOW_SHOWN
        );
        bufferSurface = SDL_CreateRGBSurfaceFrom(
            ::JD::graphics::static_rgb_buffer, 
            BUFFER_WIDTH,
            BUFFER_HEIGHT,
            24, 
            BUFFER_WIDTH * BYTES_PER_PIXEL,
            0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000
        );

        screenSurface = SDL_GetWindowSurface(window);
        viewRect.x = 0;
        viewRect.y = 0;
        viewRect.w = WINDOW_WIDTH;
        viewRect.h = WINDOW_HEIGHT;
    }
#endif

    bool quit = false;
#ifdef USE_SDL
    SDL_Event e;
#endif
    clock_t start, end;

    while (!quit) {
        auto t_frame_start = std::chrono::high_resolution_clock::now();

#ifdef USE_SDL
        if (!headless) {
            while (SDL_PollEvent(&e) != 0) {
                if (e.type == SDL_QUIT) quit = true;
                
                if (e.type == SDL_KEYDOWN) {
                    switch (e.key.keysym.sym) {
                        case SDLK_UP:    viewRect.y -= 10; break;
                        case SDLK_DOWN:  viewRect.y += 10; break;
                        case SDLK_LEFT:  viewRect.x -= 10; break;
                        case SDLK_RIGHT: viewRect.x += 10; break;
                    }
                }
            }

            if (viewRect.x < 0) viewRect.x = 0;
            if (viewRect.y < 0) viewRect.y = 0;
            if (viewRect.x + viewRect.w > BUFFER_WIDTH) viewRect.x = BUFFER_WIDTH - viewRect.w;
            if (viewRect.y + viewRect.h > BUFFER_HEIGHT) viewRect.y = BUFFER_HEIGHT - viewRect.h;

            memset(JD::graphics::static_rgb_buffer, 0, (size_t)BUFFER_HEIGHT * BUFFER_WIDTH * BYTES_PER_PIXEL);
            JD::floaters::drawFloaters();
        }
#endif

        auto t_grid_start = std::chrono::high_resolution_clock::now();
        JD::spatial::offsetsCreation();
        JD::spatial::computeIndicies();
        auto t_grid_end = std::chrono::high_resolution_clock::now();

#ifdef USE_SDL
        if (!headless) {
            SDL_BlitSurface(bufferSurface, &viewRect, screenSurface, nullptr);
            SDL_UpdateWindowSurface(window);
        }
#endif

        auto t_sim_start = std::chrono::high_resolution_clock::now();
        simulateFloaters();
        auto t_sim_end = std::chrono::high_resolution_clock::now();

        static int frame_num = 0;
        frame_num++;
        
        if (frame_num % 1 == 0) { 
           JD::logging::log(frame_num);
        }

        std::chrono::duration<double, std::milli> ms_grid = t_grid_end - t_grid_start;
        std::chrono::duration<double, std::milli> ms_sim = t_sim_end - t_sim_start;
        std::chrono::duration<double, std::milli> ms_total = std::chrono::high_resolution_clock::now() - t_frame_start;

        std::cout << "Frame " << (frame_num-1)
                  << " | Grid: " << std::fixed << std::setprecision(3) << ms_grid.count() << "ms"
                  << " | Sim: " << ms_sim.count() << "ms"
                  << " | Total: " << ms_total.count() << "ms" << std::endl;
    
        std::string frame_name = base + "/frames/";
        frame_name += std::to_string(frame_num);
        std::cout << "NAME:" <<  frame_name << '\n';
        frame_name += ".ppm";
        std::cout << "NAME:" << frame_name << '\n';

        // JD::graphics::outputPPM(BUFFER_HEIGHT, BUFFER_WIDTH, frame_name);
    }

#ifdef USE_SDL
    if (!headless) {
        SDL_FreeSurface(bufferSurface);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
#endif

     JD::logging::finish();
     return 0;
}
