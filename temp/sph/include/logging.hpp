#ifndef _LOGGING_HPP
#define _LOGGING_HPP

#include "struct.hpp"
#include "floaters.hpp"

#include <string>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <fstream>
#include <string> 
#include <sycl/sycl.hpp>

namespace JD::logging
{
    void init();
    void log(size_t i,
             int* offsets_in,
             int* cells_ctr_in,
             int* particles_loc_in, 
             int region_amt, 
             JD::floaters::block* blocks_in,
             floaters_soa particles_in,
             float h_in, 
             ::sycl::queue& q);
    void finish();
}

#endif // _LOGGING_HPP
