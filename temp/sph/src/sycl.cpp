#include "sycl.hpp"

namespace JD::sycl
{
    ::sycl::queue compute_queue{::sycl::default_selector{}};
}
