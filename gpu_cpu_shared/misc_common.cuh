#ifndef MISC_COMMON_H
#define MISC_COMMON_H

#include "defines_common.cuh"

namespace NSFem {
using real = float;

struct Point2D {
    DEVICE Point2D() : x(0), y(0) {}
    DEVICE Point2D(real x, real y) : x(x), y(y) {}
    DEVICE Point2D operator+(const Point2D& other) const {
        return Point2D(x + other.x, y + other.y);
    }
    DEVICE Point2D operator-(const Point2D& other) const {
        return Point2D(x - other.x, y - other.y);
    }
    DEVICE Point2D operator*(const real scalar) const {
        return Point2D(x * scalar, y * scalar);
    }
    /// Find the squared distance to another 2D point
    DEVICE real distToSq(const Point2D& other) const {
        return (x - other.x)*(x - other.x) + (y - other.y)*(y - other.y);
    }
    DEVICE real operator[](const int idx) const {
        assert(idx < 2);
        return (&x)[idx];
    }
    DEVICE real& operator[](const int idx) {
        assert(idx < 2);
        return (&x)[idx];
    }
    real x, y;
};

class BBox2D {
public:
    DEVICE BBox2D() :
        min(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()),
        max(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity())
    {
        static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
    }

    DEVICE BBox2D(const Point2D& min, const Point2D& max) : 
        min(min), max(max)
    {}

    DEVICE void reset() {
        min.x = std::numeric_limits<float>::infinity();
        min.y = std::numeric_limits<float>::infinity();
        max.x = -std::numeric_limits<float>::infinity();
        max.y = -std::numeric_limits<float>::infinity();
    }

    DEVICE void expand(const Point2D& point) {
        min.x = std::min(point.x, min.x);
        min.y = std::min(point.y, min.y);

        max.x = std::max(point.x, point.x);
        max.y = std::max(point.y, point.y);
    }

    DEVICE bool isInside(const Point2D& point) {
        return (min.x <= point.x && point.x <= max.x) && (min.y <= point.y && point.y <= max.y);
    }

    DEVICE const Point2D getMax() const {
        return max;
    }

    DEVICE const Point2D getMin() const {
        return min;
    }
private:
    Point2D min;
    Point2D max;
};
}

#endif
