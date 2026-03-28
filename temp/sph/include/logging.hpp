#ifndef _LOGGING_HPP
#define _LOGGING_HPP

#include <string>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <fstream>
#include <string> 

namespace JD::logging
{
    void init();
    void log(size_t i);
    void finish();
}

#endif // _LOGGING_HPP
