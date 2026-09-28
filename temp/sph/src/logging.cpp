#include "logging.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <vector>

#include "metadata.hpp"
#include "settings.hpp"
#include "sycl.hpp"

#if defined(SPH_DENSITY_ONLY) && defined(SPH_DENSITY_VELOCITY)
#error "SPH_DENSITY_ONLY and SPH_DENSITY_VELOCITY are mutually exclusive"
#endif

#if defined(SPH_DENSITY_VELOCITY)
inline constexpr JD::metadata::Variant kVariant = JD::metadata::Variant::DensityVelocity;
#elif defined(SPH_DENSITY_ONLY)
inline constexpr JD::metadata::Variant kVariant = JD::metadata::Variant::DensityOnly;
#else
#define SPH_DENSITY_ONLY
inline constexpr JD::metadata::Variant kVariant = JD::metadata::Variant::DensityOnly;
#endif

namespace JD::logging
{
    namespace
    {
        std::ofstream log_file;
        std::string logging_directory;
        float* density_frame = nullptr;

        std::string sessionName()
        {
            const auto now = std::chrono::system_clock::now();
            const std::time_t time = std::chrono::system_clock::to_time_t(now);
            std::tm local{};
#if defined(_WIN32)
            localtime_s(&local, &time);
#else
            localtime_r(&time, &local);
#endif
            std::ostringstream name;
            name << std::put_time(&local, "%Y%m%d_%H%M%S");
            std::random_device device;
            std::mt19937 generator(device());
            std::uniform_int_distribution<unsigned int> distribution(0, 0xffffu);
            name << '_' << std::hex << std::setw(4) << std::setfill('0') << distribution(generator);
            if (const char* task = std::getenv("SLURM_ARRAY_TASK_ID"); task != nullptr && *task != '\0') {
                name << "_task" << task;
            }
            return name.str();
        }

        void writeField(std::ofstream& file, const float* values, std::size_t count)
        {
            file.write(reinterpret_cast<const char*>(values), static_cast<std::streamsize>(count * sizeof(float)));
        }
    }

    bool init()
    {
        const char* root_environment = std::getenv("SPH_DATA_ROOT");
        const std::filesystem::path base = root_environment != nullptr && *root_environment != '\0' ? root_environment : "data";
        std::error_code error;
        std::filesystem::create_directories(base, error);
        if (error) {
            return false;
        }
        std::filesystem::create_directories(base / "frames", error);
        if (error) {
            return false;
        }
        logging_directory = (base / sessionName()).string();
        std::filesystem::create_directories(logging_directory, error);
        if (error) {
            return false;
        }
        const std::filesystem::path metadata_path = std::filesystem::path(logging_directory) / JD::metadata::filename;
        if (!JD::metadata::write(metadata_path, kVariant, BUFFER_WIDTH, BUFFER_HEIGHT)) {
            return false;
        }
        log_file.open(std::filesystem::path(logging_directory) / "sim_data.bin", std::ios::binary | std::ios::trunc);
        if (!log_file.is_open()) {
            return false;
        }
        density_frame = ::sycl::malloc_shared<float>(static_cast<std::size_t>(BUFFER_WIDTH) * BUFFER_HEIGHT, JD::sycl::compute_queue);
        return density_frame != nullptr;
    }

    void finish()
    {
        if (log_file.is_open()) {
            log_file.close();
        }
        if (density_frame != nullptr) {
            ::sycl::free(density_frame, JD::sycl::compute_queue);
            density_frame = nullptr;
        }
    }

    void log(std::size_t frame,
             int* offsets_in,
             int* cells_ctr_in,
             int* particles_loc_in,
             int region_amount,
             JD::floaters::block* blocks_in,
             floaters_soa particles,
             float particle_size,
             ::sycl::queue& queue)
    {
        (void)frame;
        (void)particle_size;
        if (!log_file.is_open() || offsets_in == nullptr || cells_ctr_in == nullptr || particles_loc_in == nullptr || blocks_in == nullptr) {
            return;
        }
        queue.wait();
        constexpr std::size_t pixel_count = static_cast<std::size_t>(BUFFER_WIDTH) * BUFFER_HEIGHT;
        std::vector<float> velocity_x(pixel_count, 0.0f);
        std::vector<float> velocity_y(pixel_count, 0.0f);
        std::vector<float> obstacle_mask(pixel_count, 0.0f);
        std::vector<int> velocity_count(pixel_count, 0);
        for (std::size_t index = 0; index < JD::floaters::FLOATER_AMT; ++index) {
            const float x = particles.x[index];
            const float y = particles.y[index];
            if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0f || y < 0.0f || x >= BUFFER_WIDTH || y >= BUFFER_HEIGHT) {
                continue;
            }
            const int px = static_cast<int>(x);
            const int py = static_cast<int>(y);
            if (px < 0 || px >= BUFFER_WIDTH || py < 0 || py >= BUFFER_HEIGHT) {
                continue;
            }
            const std::size_t pixel = static_cast<std::size_t>(py) * BUFFER_WIDTH + static_cast<std::size_t>(px);
            if (particles.enabled[index]) {
                velocity_x[pixel] += particles.v_x[index];
                velocity_y[pixel] += particles.v_y[index];
                ++velocity_count[pixel];
            } else {
                obstacle_mask[pixel] = 1.0f;
            }
        }
        for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
            if (velocity_count[pixel] > 0) {
                const float count = static_cast<float>(velocity_count[pixel]);
                velocity_x[pixel] /= count;
                velocity_y[pixel] /= count;
            }
        }

        if (density_frame == nullptr) {
            return;
        }
        queue.fill(density_frame, 0.0f, pixel_count).wait();
        queue.parallel_for(::sycl::range<2>(BUFFER_HEIGHT, BUFFER_WIDTH), [=](::sycl::id<2> id) {
            const int x = static_cast<int>(id[1]);
            const int y = static_cast<int>(id[0]);
            const int bx = x / DISTANCE_BETWEEN_POINTS;
            const int by = y / DISTANCE_BETWEEN_POINTS;
            float value = 0.0f;
            if (bx >= 0 && bx < BUFFER_LINE && by >= 0 && by < BUFFER_LINE) {
                const std::size_t block_index = static_cast<std::size_t>(bx + by * BUFFER_LINE);
                for (int region = 0; region < region_amount; ++region) {
                    const std::uint32_t neighbor = blocks_in[block_index].regions[region];
                    if (neighbor == 0xffffffffu) {
                        continue;
                    }
                    const int neighbor_index = static_cast<int>(neighbor);
                    if (neighbor_index < 0 || neighbor_index >= BUFFER_LINE * BUFFER_LINE) {
                        continue;
                    }
                    const int start = offsets_in[neighbor_index];
                    const int count = cells_ctr_in[neighbor_index];
                    for (int item = 0; item < count; ++item) {
                        const int particle = particles_loc_in[start + item];
                        if (particle < 0 || particle >= static_cast<int>(JD::floaters::FLOATER_AMT)) {
                            continue;
                        }
                        const float dx = particles.x[particle] - static_cast<float>(x);
                        const float dy = particles.y[particle] - static_cast<float>(y);
                        const float distance_squared = dx * dx + dy * dy;
                        if (distance_squared <= 1.0f) {
                            const float distance = ::sycl::sqrt(distance_squared);
                            value += 1.0f - distance;
                        }
                    }
                }
            }
            density_frame[static_cast<std::size_t>(y) * BUFFER_WIDTH + static_cast<std::size_t>(x)] = value;
        }).wait();

        writeField(log_file, density_frame, pixel_count);
        writeField(log_file, velocity_x.data(), pixel_count);
        writeField(log_file, velocity_y.data(), pixel_count);
        writeField(log_file, obstacle_mask.data(), pixel_count);
        log_file.flush();
    }

    const std::string& sessionDirectory()
    {
        return logging_directory;
    }

    const float* densityFrame()
    {
        return density_frame;
    }

    const char* modelVariant()
    {
        return JD::metadata::modelVariant(kVariant);
    }
}
