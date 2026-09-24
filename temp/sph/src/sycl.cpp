#include "sycl.hpp"

// Think all I do here is define queue?


namespace JD::sycl
{
    ::sycl::queue compute_queue = ::sycl::queue{::sycl::default_selector{}, ::sycl::property::queue::in_order()};

}
