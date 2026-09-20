#pragma once

#include <cstdlib>
//#include <memory>
#include <mutex>

#if defined(__has_feature)
    #if __has_feature(address_sanitizer) || __has_feature(leak_sanitizer)
        #include <sanitizer/lsan_interface.h>
        #define SA_LSAN_IGNORE(p) __lsan_ignore_object(p)
    #endif
#elif defined(__SANITIZE_ADDRESS__)
    #include <sanitizer/lsan_interface.h>
    #define SA_LSAN_IGNORE(p) __lsan_ignore_object(p)
#endif

#ifndef SA_LSAN_IGNORE
    #define SA_LSAN_IGNORE(p) ((void)0)
#endif

template <typename T>
class Singleton
{
  protected:
    static inline T*         unique_instance{nullptr};
    static inline std::mutex unique_instance_mutex;

  public:
    Singleton(const Singleton&)            = delete;
    Singleton& operator=(const Singleton&) = delete;

    static T* getInstance()
    {
        std::lock_guard lock(unique_instance_mutex);
        if (unique_instance == nullptr)
        {
            unique_instance = new T();
            SA_LSAN_IGNORE(unique_instance);
        }

        return unique_instance;
    }

    static void deleteInstance()
    {
        std::lock_guard lock(unique_instance_mutex);
        delete unique_instance;
        unique_instance = nullptr;
    }

    static T& getRef()
    {
        return *getInstance();
    }
};