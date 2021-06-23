#pragma once
#include "error_code.h"
#include <chrono>

namespace Profiling {
    class Timer {
    public: 
    Timer() :
        start(std::chrono::steady_clock::now())
    {

    }

    void reset() {
        start = std::chrono::steady_clock::now();
    }

    float getDurationMS() {
        return getDuration<float, std::milli>();
    }

    float getDurationS() {
        return getDuration<float, std::ratio<1, 1>>();
    }

    float getDurationMSReset() {
        return getDurationReset<float, std::milli>();
    }

    float getDurationSReset() {
        return getDurationReset<float, std::ratio<1, 1>>();
    }

    protected:
        std::chrono::time_point<std::chrono::steady_clock> start;
        template<typename T, typename Ratio>
        T getDuration() {
            const std::chrono::duration<T, Ratio> duration = std::chrono::steady_clock::now() - start;
            return duration.count();
        }

        template<typename T, typename Ratio>
        T getDurationReset() {
            auto now = std::chrono::steady_clock::now();
            const std::chrono::duration<T, Ratio> duration = now - start;
            start = now;
            return duration.count();
        }
    };

    #ifdef _MSC_VER
        #define TIMER_FUNCTION_NAME __FUNCSIG__
    #else
        #define TIMER_FUNCTION_NAME __PRETTY_FUNCTION__
    #endif

    #define _PROFILING_COMBINE_NAME_HELPER(A, B) A##B
    #define _PROFILING_COMBINE_NAME(A, B) _PROFILING_COMBINE_NAME_HELPER(A, B) 

#ifdef WITH_PROFILING_SCOPED_TIMERS
    #define PROFILING_SCOPED_TIMER_FUN() Profiling::ScopedTimer _PROFILING_COMBINE_NAME(__scopedTimer, __LINE__)(TIMER_FUNCTION_NAME);
    // concatiname with empty string literal to assure that ARG is string literal
    #define _PROFILING_SCOPED_TIMER_CUSTOM_HELPER(ARG) ARG ""
    #define PROFILING_SCOPED_TIMER_CUSTOM(ARG) Profiling::ScopedTimer _PROFILING_COMBINE_NAME(__scopedTimer, __LINE__)(PROFILING_SCOPED_TIMER_CUSTOM_HELPER(ARG));
#else
    #define PROFILING_SCOPED_TIMER_FUN()
    #define PROFILING_SCOPED_TIMER_CUSTOM(ARG) 
#endif
    class ScopedTimer : public Timer {
    public:
        ScopedTimer() :
            Timer(),
            name(nullptr)
        {

        }
        /// Use this constructor only with string literals. It's recommended to use the provided macros
        /// SCOPED_TIMER_FUN, SCOPED_TIMER_CUSTOM
        ScopedTimer(const char* name) : 
            Timer(),
            name(name)
        {

        }
        ~ScopedTimer() {
            if(name) {
                printf("[Scoped Timer][%s] %fs\n", name, getDurationS());
            } else {
                printf("[Scoped Timer][__anonymous__] %fs\n", getDurationS());
            }
        }
    private:
        const char* name;
    };
};