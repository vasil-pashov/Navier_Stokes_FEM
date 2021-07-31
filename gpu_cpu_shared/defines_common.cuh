#ifndef GPU_CPU_DEFINES_H
#define GPU_CPU_DEFINES_H

#ifdef __CUDACC__
    #define DEVICE __device__
#else
    #define DEVICE
#endif

#if defined _MSC_VER
	#define FORCEINLINE __forceinline
#else
	#define FORCEINLINE __attribute__ ((always_inline)) inline
#endif

namespace NSFemGPU {

template<typename T>
DEVICE FORCEINLINE T max(const T a, const T b) {
    return a > b ? a : b;
}

template<typename T>
DEVICE FORCEINLINE T min(const T a, const T b) {
    return a < b ? a : b;
}

}

#endif
