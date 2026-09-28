#ifndef JD_LOGGING_HPP
#define JD_LOGGING_HPP

#include <cstddef>
#include <string>

#include <sycl/sycl.hpp>

#include "floaters.hpp"
#include "struct.hpp"

namespace JD::logging
{
    bool init();
    void finish();
    void log(std::size_t frame,
             int* offsets_in,
             int* cells_ctr_in,
             int* particles_loc_in,
             int region_amount,
             JD::floaters::block* blocks_in,
             floaters_soa particles,
             float particle_size,
             ::sycl::queue& queue);
    const std::string& sessionDirectory();
    const float* densityFrame();
    const char* modelVariant();
}

#endif
