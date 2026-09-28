#ifndef JD_STRUCT_HPP
#define JD_STRUCT_HPP

#include <cstdint>

#include "settings.hpp"

struct SpawnBox
{
    float x;
    float y;
    float w;
    float h;
};

struct point
{
    std::uint16_t x;
    std::uint16_t y;
    std::uint16_t i_x;
    std::uint16_t i_y;
    std::uint16_t id;
    float strength;
    int regions[REGIONS_AMT];
};

struct floater
{
    float density;
    float p_x;
    float p_y;
    float x;
    float y;
    float v_x;
    float v_y;
    float v_x_h;
    float v_y_h;
    float a_x;
    float a_y;
    float mass;
    float pressure;
    bool enabled;
};

struct floaters_soa
{
    float* density;
    float* p_x;
    float* p_y;
    float* x;
    float* y;
    float* v_x;
    float* v_y;
    float* v_x_h;
    float* v_y_h;
    float* a_x;
    float* a_y;
    float* mass;
    float* pressure;
    bool* enabled;
};

struct force
{
    float x;
    float y;
};

#endif
