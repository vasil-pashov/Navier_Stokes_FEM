#pragma once
#include<cstring>
#include<cassert>

namespace NSFem {

/// Class to represent matrix allocated on the stack. It has various operations related to matrices
/// such as: matrix multiplication, addition, subtraction, finding determinant, etc.
template<typename T, int rows, int cols>
class StaticMatrix {
public:
    StaticMatrix() {
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                matrix[i][j] = T(0);
            }
        }
    }

    explicit StaticMatrix(T in[rows][cols]) : StaticMatrix() {
        std::memcpy(matrix, in, sizeof(T) * rows * cols);
    }

    explicit StaticMatrix(T (&&in)[rows][cols]) : StaticMatrix() {
        std::memcpy(matrix, in, sizeof(T) * rows * cols);
    }

    explicit StaticMatrix(T diagonal) : StaticMatrix() {
        static_assert(rows == cols, "This constructor is used to set the main diagonal of square matrix.");
        memset(&matrix[0][0], 0, sizeof(T) * rows * cols);
        for (int i = 0; i < rows; ++i) matrix[i][i] = diagonal;
    }

    using Iterator = T*;
    Iterator begin() {
        return data();
    }

    Iterator end() {
        return data() + rows * cols;
    }

    const T* data() const {
        return &matrix[0][0];
    }

    T* data() {
        return &matrix[0][0];
    }

    constexpr int getRows() const {
        return rows;
    }

    constexpr int getCols() const {
        return cols;
    }

    template<int otherCols>
    const StaticMatrix<T, rows, otherCols> operator*(const StaticMatrix<T, cols, otherCols>& rhs) const {
        StaticMatrix<T, rows, otherCols> result;
        memset(result.data(), 0, sizeof(T) * rows * otherCols);
        for (int k = 0; k < otherCols; ++k) {
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) {
                    result.element(row, k) += (*this).element(row, col) * rhs.element(col, k);
                }
            }
        }
        return result;
    }

    const StaticMatrix<T, rows, cols> operator*(const T rhs) const {
        StaticMatrix<T, rows, cols> result;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.element(i, j) = element(i, j) * rhs;
            }
        }
        return result;
    }

    const StaticMatrix<T, rows, cols> operator/(const T rhs) const {
        return (*this) * (T(1) / rhs);
    }

    const StaticMatrix<T, rows, cols> operator+(const StaticMatrix<T, rows, cols>& rhs) const {
        StaticMatrix<T, rows, cols> result;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.element(i, j) = element(i, j) + rhs.data[i][j];
            }
        }
    }

    const StaticMatrix<T, rows, cols> operator-(const StaticMatrix<T, rows, cols>& rhs) const {
        StaticMatrix<T, rows, cols> result;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.element(i, j) = element(i, j) - rhs.data[i][j];
            }
        }
    }

    void operator +=(const StaticMatrix<T, rows, cols>& other) {
         for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                element(i, j) += other.element(i, j);
            }
        }
    }

    T* operator[](const int i) {
        assert(i < rows);
        return &(matrix[i][0]);
    }

    const T* operator[](const int i) const {
        assert(i < rows);
        return &(matrix[i][0]);
    }

    T& element(const int i, const int j) {
        assert(0 <= i && i < rows && 0 <= j && j < cols);
        return matrix[i][j];
    }

    const T element(const int i, const int j) const {
        assert(i < rows && j < cols);
        return matrix[i][j];
    }

    const T getDet() const {
        return det(*this);
    }

    const StaticMatrix<T, cols, rows> getTransposed() const {
        StaticMatrix<T, cols, rows> result;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.element(j, i) = element(i, j);
            }
        }
        return result;
    }

private:
    T matrix[rows][cols];
};

}

