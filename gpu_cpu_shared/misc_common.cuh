#ifndef MISC_COMMON_H
#define MISC_COMMON_H

#include "defines_common.cuh"

namespace NSFem {
using real = float;

struct Point2D {
    device Point2D() : x(0), y(0) {}
    device Point2D(real x, real y) : x(x), y(y) {}
    device Point2D operator+(const Point2D& other) const {
        return Point2D(x + other.x, y + other.y);
    }
    device Point2D operator-(const Point2D& other) const {
        return Point2D(x - other.x, y - other.y);
    }
    device Point2D operator*(const real scalar) const {
        return Point2D(x * scalar, y * scalar);
    }
    /// Find the squared distance to another 2D point
    device real distToSq(const Point2D& other) const {
        return (x - other.x)*(x - other.x) + (y - other.y)*(y - other.y);
    }
    device real operator[](const int idx) const {
        assert(idx < 2);
        return (&x)[idx];
    }
    device real& operator[](const int idx) {
        assert(idx < 2);
        return (&x)[idx];
    }
    real x, y;
};

class BBox2D {
public:
    device BBox2D() :
        min(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()),
        max(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity())
    {
        static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
    }

    device BBox2D(const Point2D& min, const Point2D& max) : 
        min(min), max(max)
    {}

    device void reset() {
        min.x = std::numeric_limits<float>::infinity();
        min.y = std::numeric_limits<float>::infinity();
        max.x = -std::numeric_limits<float>::infinity();
        max.y = -std::numeric_limits<float>::infinity();
    }

    device void expand(const Point2D& point) {
        min.x = std::min(point.x, min.x);
        min.y = std::min(point.y, min.y);

        max.x = std::max(point.x, point.x);
        max.y = std::max(point.y, point.y);
    }

    device bool isInside(const Point2D& point) {
        return (min.x <= point.x && point.x <= max.x) && (min.y <= point.y && point.y <= max.y);
    }

    device const Point2D getMax() const {
        return max;
    }

    device const Point2D getMin() const {
        return min;
    }
private:
    Point2D min;
    Point2D max;
};
}

#endif
