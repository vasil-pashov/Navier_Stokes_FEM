#ifndef GPU_CPU_DEFINES_H
#define GPU_CPU_DEFINES_H

#ifdef __CUDACC__
    #define device __device__
#else
    #define device
#endif

#endif
