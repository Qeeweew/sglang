#pragma once

// Shared cold-compile exclusion. DiskCache still owns lookup and publication.
#include <cerrno>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace deep_jit {

class CompileLock {
    int fd_ = -1;

  public:
    CompileLock(const std::filesystem::path &cache_root, const std::string &digest) {
        const auto directory = cache_root / "compile_locks";
        std::filesystem::create_directories(directory);
        const auto path = directory / (digest + ".lock");
        fd_ = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0664);
        if (fd_ < 0)
            throw std::system_error(errno, std::generic_category(), path.string());
        while (::flock(fd_, LOCK_EX) != 0) {
            if (errno == EINTR)
                continue;
            const int error = errno;
            ::close(fd_);
            throw std::system_error(error, std::generic_category(), path.string());
        }
    }

    CompileLock(const CompileLock &) = delete;
    CompileLock &operator=(const CompileLock &) = delete;
    // Keep the inode: unlinking can let a new caller bypass existing waiters.
    // close also releases the lock after exceptions; the OS releases it on exit.
    ~CompileLock() { ::close(fd_); }
};

} // namespace deep_jit
