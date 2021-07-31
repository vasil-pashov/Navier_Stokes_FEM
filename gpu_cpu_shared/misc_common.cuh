#ifndef MISC_COMMON_H
#define MISC_COMMON_H

#include "defines_common.cuh"
#include <float.h>

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
        min(FLT_MAX, FLT_MAX),
        max(-FLT_MAX, -FLT_MAX)
    {

    }

    DEVICE BBox2D(const Point2D& min, const Point2D& max) : 
        min(min), max(max)
    {
        
    }

    DEVICE void reset() {
        min.x = FLT_MAX;
        min.y = FLT_MAX;
        max.x = -FLT_MAX;
        max.y = -FLT_MAX;
    }

    DEVICE void expand(const Point2D& point) {
        min.x = NSFemGPU::min(point.x, min.x);
        min.y = NSFemGPU::min(point.y, min.y);

        max.x = NSFemGPU::max(point.x, point.x);
        max.y = NSFemGPU::max(point.y, point.y);
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
