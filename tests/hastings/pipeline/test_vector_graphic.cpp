#include <gtest/gtest.h>

#include "hastings/pipeline/vector_graphic.h"

TEST(VectorGraphic, LineGraphic) {
    using hastings::LineGraphic;
    using hastings::VectorGraphic;

    VectorGraphic graphic{LineGraphic{{0, 255, 0}, {10, 20}, {30, 40}}};
}

TEST(VectorGraphic, PointGraphic) {
    using hastings::PointGraphic;
    using hastings::VectorGraphic;

    VectorGraphic graphic{PointGraphic{{0, 255, 0}, {25, 35}}};
}

TEST(VectorGraphic, RectangleGraphic) {
    using hastings::RectangleGraphic;
    using hastings::VectorGraphic;

    VectorGraphic graphic{RectangleGraphic{{0, 255, 0}, {25, 35}, {45, 55}}};
}

TEST(VectorGraphic, TextGraphic) {
    using hastings::TextGraphic;
    using hastings::VectorGraphic;

    VectorGraphic graphic{TextGraphic{{0, 255, 0}, {25, 35}, "hello world"}};
}