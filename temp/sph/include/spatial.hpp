#ifndef JD_SPATIAL_HPP
#define JD_SPATIAL_HPP

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace JD::spatial
{
    void offsetsCreation();
    std::vector<std::pair<int, int>> calculateRegionsOffsets();
    void computeIndicies();
    void computeBlockIndicies();
    void rebuild();
    std::optional<std::size_t> cellIndex(float x, float y);
}

#endif
