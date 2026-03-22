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
    
    // Ok as modular & customisable as I want to make this
    // I need to be realisitc with how much time I can spend onthis
    // I have like 2 weeks, so I will hard code parts of it, i.e
    // I will just write down how the log function works, and init function works
    // They are not meant to be used long term

    void log(size_t i);
}

#endif // _LOGGING_HPP
