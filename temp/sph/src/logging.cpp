#include "logging.hpp"

namespace JD::logging
{
    static std::string _logging_dir;

    // From gemini
    static std::string _get_log_dirname() {
        // 1. Get current time (UTC)
        /*
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm bt = *std::gmtime(&now_time); // Use UTC to avoid timezone headache

        // 2. Format the timestamp: YYYYMMDD_HHMMSS
        std::ostringstream oss;
        oss << std::put_time(&bt, "%Y%m%d_%H%MSS");

        // 3. Generate a 4-character random hex string
        // We use random_device to seed a Mersenne Twister
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 0xFFFF); // 2 bytes of randomness
        
        oss << "_" << std::hex << std::setw(4) << std::setfill('0') << dis(gen);
*/
        return "qq";
    }

    // This function makes a logging dir incase there is not already one
    void init()
    {
        _logging_dir = _get_log_dirname();
        if (!std::filesystem::is_directory(_logging_dir) || !std::filesystem::exists(_logging_dir)) 
        {
            std::filesystem::create_directory(_logging_dir);
        }
    }
    
    // Ok as modular & customisable as I want to make this
    // I need to be realisitc with how much time I can spend onthis
    // I have like 2 weeks, so I will hard code parts of it, i.e
    // I will just write down how the log function works, and init function works
    // They are not meant to be used long term

    // idfk how the hell this is gonna work
    // I am gonna 
    // ok wait I was gonna say vibe code
    // but it now means smth else...
    // but Im just gonna turn off my mind and slowly write all of it
    void log(size_t i)
    {
        // file format = i_[d/v_x/v_y/m].csv
        // NxN entries for each pixel in the position
        // with float to 3 decimals of percision
        // mask is just 1/0

        /*
        // density calc
        {
            std::ofstream density_log(density_path);
            if (!density_log.is_open()) {
                throw std::runtime_error("Could not open file: " + density_path);
            }
            density_log << "Initialising system..." << std::endl; // std::endl flushes automatically
            

            density_log.close();
        }
        */
    }

}

