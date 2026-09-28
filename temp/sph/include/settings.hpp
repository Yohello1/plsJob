#ifndef JD_SETTINGS_HPP
#define JD_SETTINGS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>

inline constexpr int DISTANCE_BETWEEN_POINTS = 8;
inline constexpr int SIZE_MULTIPLIER = 40;
inline constexpr int INFLUENCE_RADIUS = 4;
inline constexpr int PADDING = INFLUENCE_RADIUS + 1;

inline constexpr int BUFFER_WIDTH = SIZE_MULTIPLIER * DISTANCE_BETWEEN_POINTS + DISTANCE_BETWEEN_POINTS * PADDING * 2;
inline constexpr int BUFFER_HEIGHT = SIZE_MULTIPLIER * DISTANCE_BETWEEN_POINTS + DISTANCE_BETWEEN_POINTS * PADDING * 2;
inline constexpr int BUFFER_PADDING = PADDING * DISTANCE_BETWEEN_POINTS;
inline constexpr int BUFFER_UNPADDED = BUFFER_PADDING + SIZE_MULTIPLIER * DISTANCE_BETWEEN_POINTS;
inline constexpr int BUFFER_LINE = PADDING * 2 + SIZE_MULTIPLIER;
inline constexpr int BUFFER_WORKING = SIZE_MULTIPLIER * DISTANCE_BETWEEN_POINTS;
inline constexpr int BYTES_PER_PIXEL = 3;
inline constexpr int SCREEN_SCALE = 1;
inline constexpr int POINTS_WIDTH = SIZE_MULTIPLIER + 1;
inline constexpr int POINTS_HEIGHT = SIZE_MULTIPLIER + 1;
inline constexpr int POINTS_AMT = POINTS_WIDTH * POINTS_HEIGHT;
inline constexpr float THRESHOLD = 0.05f;

inline constexpr int WINDOW_WIDTH = 640;
inline constexpr int WINDOW_HEIGHT = 640;

inline constexpr float PARTICLE_SIZE = 3.0f;
inline constexpr float PARTICLE_TIME_STEP = 0.10f;
inline constexpr float PARTICLE_REFERENCE_DENSITY = 0.030f;
inline constexpr float PARTICLE_BULK_MODULUS = 2000.0f;
inline constexpr float PARTICLE_VISCOSITY = 0.5f;
inline constexpr float PARTICLE_GRAVITY = 10.0f;
inline constexpr float PARTICLE_MASS = 0.015f;
inline constexpr float PARTICLE_REPULSION = 0.5f * PARTICLE_BULK_MODULUS;
inline constexpr float PARTICLE_MAX_V = 7.5f;
inline constexpr float PARTICLE_RESTITUTION = 1.0f;
inline constexpr int PARTICLE_GHOST_DENSITY = 1;
inline constexpr int PARTICLE_N_FRAMES = 1;
inline constexpr int PARTICLE_NP_FRAMES = 1;

inline constexpr float PI = std::numbers::pi_v<float>;
inline constexpr float PARTICLE_VISCOSITY_K_COEFF = 25.0f / PI;
inline constexpr float PARTICLE_SPIKY_K = -45.0f / (PI * PARTICLE_SIZE * PARTICLE_SIZE * PARTICLE_SIZE * PARTICLE_SIZE * PARTICLE_SIZE * PARTICLE_SIZE);
inline constexpr float PARTICLE_POLY6_K_SMOOTHING = 315.0f / (64.0f * PI * PARTICLE_SIZE * PARTICLE_SIZE * PARTICLE_SIZE * PARTICLE_SIZE * PARTICLE_SIZE * PARTICLE_SIZE * PARTICLE_SIZE * PARTICLE_SIZE * PARTICLE_SIZE);

inline constexpr std::size_t DESIRED_FLOATERS = 50000;
inline constexpr std::size_t GHOST_FLOATERS = 200000;
inline constexpr std::size_t FLOATER_AMT = DESIRED_FLOATERS + GHOST_FLOATERS;
inline constexpr int FLOATER_SPEED = 3;

inline constexpr int CELL_SIZE = DISTANCE_BETWEEN_POINTS * DISTANCE_BETWEEN_POINTS;
inline constexpr int REGIONS_AMT = 2 * INFLUENCE_RADIUS * INFLUENCE_RADIUS - 2 * INFLUENCE_RADIUS + 1;

#endif
