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

void simulateFloaters()
{
    JD::simulate::computeDensity<JD::Poly6_k::smoothing>(
        JD::graphics::offsets,
        JD::graphics::cells_ctr,
        JD::graphics::particles_loc,
        JD::floaters::BLOCK_NEIGHBOR_COUNT,
        JD::floaters::blocks,
        JD::floaters::floatersA,
        PARTICLE_SIZE);
    JD::simulate::computePressureForce<JD::Spiky_k::gradient>(
        JD::graphics::offsets,
        JD::graphics::cells_ctr,
        JD::graphics::particles_loc,
        JD::floaters::floatersA,
        PARTICLE_SIZE);
    JD::simulate::computeViscosity<JD::Viscosity_k::laplacian>(
        JD::graphics::offsets,
        JD::graphics::cells_ctr,
        JD::graphics::particles_loc,
        JD::floaters::floatersA,
        PARTICLE_SIZE);
    JD::simulate::applyYAccelerationToAllParticles<JD::gravity::gravityAcceleration>(
        JD::floaters::floatersA);
    JD::simulate::integrate(
        JD::graphics::offsets,
        JD::graphics::cells_ctr,
        JD::graphics::particles_loc,
        JD::floaters::floatersA);
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
    std::filesystem::create_directories("data/frames");
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

        JD::graphics::InitializeStaticBuffer();

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
        start = clock();

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

        JD::spatial::offsetsCreation();
        JD::spatial::computeIndicies();

#ifdef USE_SDL
        if (!headless) {
            SDL_BlitSurface(bufferSurface, &viewRect, screenSurface, nullptr);
            SDL_UpdateWindowSurface(window);
        }
#endif

        simulateFloaters();

        static int frame_num = 0;
        if (max_frames > 0 && frame_num >= max_frames) quit = true;

        if (frame_num % 1 == 0) { 
           JD::logging::log(frame_num);
        }
        
        frame_num++;

        end = clock();
        double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
        std::cout << "Frame " << (frame_num-1)
                  << " Time: " << std::fixed << std::setprecision(4) << time_taken << "s" << std::endl;
    
        std::string frame_name = "data/frames/";
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

     return 0;
}
