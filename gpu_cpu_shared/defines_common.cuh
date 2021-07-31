#ifndef GPU_CPU_DEFINES_H
#define GPU_CPU_DEFINES_H

#ifdef __CUDACC__
    #define DEVICE __device__
#else
    #define DEVICE
#endif

#endif
