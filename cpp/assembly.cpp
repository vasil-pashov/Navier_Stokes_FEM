#include "assembly.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace NSFem {

/// OpenCV uses colors in BGR format so we make a simple wrapper to create an OpenCV color
/// in the right format
static inline cv::Scalar color(uint8_t red, uint8_t green, uint8_t blue) {
    return cv::Scalar(blue, green, red);
}


/// Interpolate the value x between start and end and produce a color. The color represents
/// heatmap, where cold colors correspond to small values and warm colors correspond to large values
/// @param[in] x Value between start and end for which we want to interpolate the color
/// @param[in] start The start of the interpolation interval
/// @param[in] end The end of the interpolation interval
static cv::Scalar heatmap(const real x, const real start, const real end) {
    const int numColors = 4;
    const cv::Scalar colors[numColors] = {
        color(0, 0, 255), // blue
        color(0, 255, 0), // green
        color(255, 255, 0), // yellow
        color(255, 0, 0) // red
    };
    const real h = (end - start) / (numColors - 1);
    if(std::abs(h) < 1e-4) {
        return colors[0];
    }
    cv::Scalar result(0, 0 ,0);
    // Lagrangian interpolation
    for(int i = 0; i < numColors; ++i) {
        real mult = 1;
        const real xi = start + i * h;
        for(int j = 0; j < numColors; ++j) {
            if(j != i) {
                const real xj = start + j * h;
                mult *= (xj - x) / (xj - xi);
            }
        }
        result += mult * colors[i];
    }
    return result;
}

float findSmallestSide(const FemGrid2D& grid) {
    const int numElements = grid.getElementsCount();
    assert(grid.getElementSize() == 6);
    int idx[6];
    Point2D nodes[6];
    float res = std::numeric_limits<float>::infinity();
    for(int i = 0; i < numElements; ++i) {
        grid.getElement(i, idx, reinterpret_cast<NSFem::real*>(nodes));
        const float minSide = std::min(
            std::min(nodes[0].distToSq(nodes[1]), nodes[0].distToSq(nodes[2])),
            nodes[1].distToSq(nodes[2]));
        res = std::min(res, minSide);
    }
    return sqrt(res);
}

void drawVectorPlot(
    cv::Mat& outputImage,
    const FemGrid2D& grid,
    const SMM::real* const uVec,
    const SMM::real* const vVec,
    const SMM::real* const pressure,
    const std::string& path,
    const int width,
    const int height,
    const float maxArrowLength
) {

    outputImage.setTo(cv::Scalar(255, 255, 255));
    // First find the maximal length, it will be used to as an end interval during heatmap
    // interpolation. Then find the max and min coordinate in x and y directions, this is
    // used when drawing the image. We want to keep the aspect ratio when drawing the grid
    // even when the image width and heigh have different aspect ratio. This means that the
    // mapping between the sim region and the image won't be 1:1.
    const real* nodes = grid.getNodesBuffer();
    real minX = nodes[0];
    real maxX = nodes[0];
    real minY = nodes[1];
    real maxY = nodes[1];
    real maxLenSq = 0;
    const int numNodes = grid.getNodesCount();
    for(int i = 0; i < numNodes; ++i) {
        const real current = uVec[i] * uVec[i] + vVec[i] * vVec[i];
        maxLenSq = std::max(maxLenSq, current);

        minX = std::min(minX, nodes[2 * i]);
        maxX = std::max(maxX, nodes[2 * i]);

        minY = std::min(minY, nodes[2 * i + 1]);
        maxY = std::max(maxY, nodes[2 * i + 1]);
    }

    const real xWidth = maxX - minX;
    const real yWidth = maxY - minY;
    const real ar = xWidth / yWidth;

    // Find the proper scaling factor so the grid in image space has the same proportions
    // as in world space
    const real xScale = (width / xWidth);
    const real yScale = width / (ar * yWidth);

    // We want to keep the image in the center of the image space. For example long but thin
    // region in world space, will match long and thin region in image space, and it will be
    // at the top of the image, so wi find the empty space divide it by 2 (so that it has the 
    // same amount of empty space above and below) and offset the image with this amount.
    const real yOffset = 0.5 * std::max(real(0), height - yScale * yWidth);
    const real xOffset = 0.5 * std::max(real(0), width - xScale * xWidth);

    real maxPressure = -std::numeric_limits<float>::infinity();
    for(int i = 0; i < grid.getPressureNodesCount(); ++i) {
        maxPressure = std::max(pressure[i], maxPressure);
    }

    const real maxLength = std::sqrt(maxLenSq);
    real maxU = 0;
    real maxV = 0;
    for(int i = 0; i < numNodes; ++i) {
        real maxU = uVec[0];
        real maxV = vVec[0];
        const real length = sqrt(uVec[i] * uVec[i] + vVec[i] * vVec[i]);
        const real lengthScaled = length * (maxArrowLength / maxLength);
        const real x = nodes[2 * i];
        const real y = nodes[2 * i + 1];
        const real xEnd = x + lengthScaled * uVec[i];
        const real yEnd = y + lengthScaled * vVec[i];

        const real xImageSpace = xOffset + x * xScale - xScale * minX;
        const real yImageSpace = yOffset + y * yScale - yScale * minY;

        const real xEndImageSpace = xOffset + xEnd * xScale - xScale * minX;
        const real yEndImageSpace = yOffset + yEnd * yScale  -yScale * minY;

        maxU = std::max(maxU, (real)uVec[i]);
        maxV = std::max(maxV, (real)vVec[i]);

        // Draw the velocity
        cv::arrowedLine(
            outputImage,
            cv::Point(xImageSpace, yImageSpace),
            cv::Point(xEndImageSpace, yEndImageSpace),
            heatmap(lengthScaled, 0, maxArrowLength)
        );

        // Draw each grid point
        const cv::Point gridPointImageSpace(
            xOffset + nodes[2 * i] * xScale - xScale * minX,
            yOffset + nodes[2* i + 1] * yScale - minY * yScale
        );
        cv::circle(
            outputImage,
            gridPointImageSpace,
            0,
            cv::Scalar(0, 0, 0),
            2
        );
    }

    for(int i = 0; i < grid.getElementsCount(); ++i) {
        int idx[6];
        Point2D nds[6];
        grid.getElement(i, idx, reinterpret_cast<real*>(nds));
        real nMinX = std::min(nds[2].x, std::min(nds[0].x, nds[1].x));
        real nMaxX = std::max(nds[2].x, std::max(nds[0].x, nds[1].x));
        real nMinY = std::min(nds[2].y, std::min(nds[0].y, nds[1].y));
        real nMaxY = std::max(nds[2].y, std::max(nds[0].y, nds[1].y));


        const real xImageSpace = nMinX * xScale + xScale * minX;
        const real yImageSpace = nMinY * yScale + yScale * minY;

        const real xEndImageSpace = nMaxX * xScale + xScale * minX;
        const real yEndImageSpace = nMaxY * yScale + yScale * minY;

        for(int row = xImageSpace; row < xEndImageSpace; ++row) {
            for(int col = yImageSpace; col < yEndImageSpace; ++col) {
                const real wx = (row + xScale * minX) / xScale;
                const real wy = (col + yScale * minY) / yScale;
                real xi, eta;
                if(isPointInTriagle(Point2D(wx, wy), nds[0], nds[1], nds[2], xi, eta)) {
                    real coeffs[3];
                    P1::eval(xi, eta, coeffs);
                    real val = coeffs[0] * pressure[idx[0]] + coeffs[1] * pressure[idx[1]] + coeffs[2] * pressure[idx[2]];
                    auto color = heatmap(val, 0, maxPressure);
                    cv::circle(
                        outputImage,
                        cv::Point(row, col),
                        1,
                        color,
                        2
                    );
                }
            }
        }
    }

    // Info string with the maximal velocity, and maximal velocity component in each direction.
    const std::string& info = "Max velocity length: " + std::to_string(maxLength) + ". Max u: " + std::to_string(maxU) + ". Max v: " + std::to_string(maxV);

    cv::putText(outputImage, info.c_str(), cv::Point(0, height), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0), 1);

    cv::imwrite(path.c_str(), outputImage);
}
}

