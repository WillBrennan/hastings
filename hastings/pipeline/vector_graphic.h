#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <variant>
#include <vector>

#include "hastings/math/math.h"

namespace hastings {

struct Graphic {
    using Color = Vec3u;
    using Pixel = Vec2f;
};

struct PointGraphic {
    Graphic::Color color;
    Graphic::Pixel point;
};

struct LineGraphic {
    Graphic::Color color;
    Graphic::Pixel start;
    Graphic::Pixel end;
};

struct RectangleGraphic {
    Graphic::Color color;
    Graphic::Pixel topLeft;
    Graphic::Pixel bottomRight;
};

struct TextGraphic {
    Graphic::Color color;
    Graphic::Pixel point;
    std::string text;
};

struct VectorGraphic {
    VectorGraphic() = default;
    VectorGraphic(PointGraphic&& g) : graphic(g) {}
    VectorGraphic(LineGraphic&& g) : graphic(g) {}
    VectorGraphic(RectangleGraphic&& g) : graphic(g) {}
    VectorGraphic(TextGraphic&& g) : graphic(g) {}

    std::variant<PointGraphic, LineGraphic, RectangleGraphic, TextGraphic> graphic;
};

using VectorGraphics = std::vector<VectorGraphic>;
}  // namespace hastings
