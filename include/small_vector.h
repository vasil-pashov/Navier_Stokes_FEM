#pragma once
namespace NSFem {

/// Class which has similar interface to std::vector, but is stored on the stack
/// @tparam T The type of the data which is going to be in the vector
/// @tparam capacity The maximum elements which the vector can hold.
template<typename T, int capacity>
class SmallVector {
public:
    SmallVector() : 
        firstFreePosition(0)
    {}

    SmallVector(const SmallVector&) = delete;
    SmallVector& operator=(const SmallVector&) = delete;
    SmallVector(SmallVector&&) = delete;
    SmallVector& operator=(SmallVector&&) = delete;

    /// Random access iterator
    using Iterator = T*;
    /// Const random access iterator
    using ConstIterator = const T*;

    /// @returns The number of used elements
    int size() const {
        return firstFreePosition;
    }

    /// Add new element at the end of the vector. The content is copied.
    /// @param[in] el The element which will be added to the vector
    void pushBack(const T& el) {
        assert(firstFreePosition < capacity);
        data[firstFreePosition] = el;
        firstFreePosition++;
    }

    /// Add new element at the end of the vector. The content is moved.
    /// @param[in] el The element which will be added to the vector
    void pushBack(T&& el) {
        assert(firstFreePosition < capacity);
        data[firstFreePosition] = std::move(el);
        firstFreePosition++;
    }

    /// Add new element at the end of the vector. The element is constructed using the
    /// passed arguments, which are forwared to the consructor.
    template<typename... ArgsT>
    void emplaceBack(ArgsT&&... args) {
        assert(firstFreePosition < capacity);
        data[firstFreePosition] = T(std::forward<ArgsT>(args)...);
        firstFreePosition++;
    }

    /// Free the last element of the vector. The element is not destryed when this is called.
    void popBack() {
        assert(firstFreePosition > 0);
        firstFreePosition = std::max(firstFreePosition - 1, 0);
    }

    /// Check of the vector is empty
    /// @returns true if there are no elements in the vector i.e. SmallVector::size() == 0
    bool empty() const {
        return firstFreePosition == 0;
    }

    /// @returns reference to the last element in the vector.
    T& back() {
        assert(firstFreePosition > 0);
        return data[firstFreePosition - 1];
    }

    /// @returns reference to the last element in the vector.
    const T& back() const {
        assert(firstFreePosition > 0);
        return data[firstFreePosition - 1];
    }

    /// @returns reference to the first element in the vector.
    T& front() {
        assert(firstFreePosition > 0);
        return data[0];
    }

    /// @returns reference to the first element in the vector.
    const T& front() const {
        assert(firstFreePosition > 0);
        return data[0];
    }

    /// @returns iterator to the beggining of the vector
    Iterator begin() {
        return data;
    }

    /// @returns iterator to one element past the end of the vector. Should not be dereferenced.
    Iterator end() {
        return data + firstFreePosition;
    }

    /// @returns constant iterator to the beggining of the vector
    ConstIterator begin() const {
        return data;
    }

    /// @returns constant iterator to one element past the end of the vector. Should not be dereferenced.
    ConstIterator end() const {
        return data + firstFreePosition;
    }

private:
    int firstFreePosition;
    T data[capacity];
};

} //NSFem
