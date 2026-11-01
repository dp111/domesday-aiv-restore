/*
 * File:        subprocess_runner.cpp
 * Module:      orc-plugin_domesday-model-decoder
 * Purpose:     Run a child process, streaming its merged stdout/stderr lines
 *              (POSIX only)
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "subprocess_runner.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/poll.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>

namespace domesday {

SubprocessResult run_subprocess(
    const std::vector<std::string>& argv,
    const std::function<void(const std::string&)>& on_line,
    const std::atomic<bool>* cancel) {
  SubprocessResult result;
  if (argv.empty()) {
    result.error = "empty command line";
    return result;
  }

  int fds[2];
  if (pipe2(fds, O_CLOEXEC) != 0) {
    result.error = std::string("pipe: ") + std::strerror(errno);
    return result;
  }

  // fork/exec rather than posix_spawn: the SDK include gate allows only
  // unistd.h / fcntl.h / sys/* platform headers.  A CLOEXEC pipe reports an
  // exec failure back to the parent.
  int err_fds[2];
  if (pipe2(err_fds, O_CLOEXEC) != 0) {
    close(fds[0]);
    close(fds[1]);
    result.error = std::string("pipe: ") + std::strerror(errno);
    return result;
  }
  std::vector<char*> cargv;
  for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
  cargv.push_back(nullptr);

  const pid_t pid = fork();
  if (pid == 0) {
    // Child: only async-signal-safe calls from here.  Own process group, so
    // a cancel also reaches ld-chroma-decoder and the decoder's workers.
    setpgid(0, 0);
    const int devnull = open("/dev/null", O_RDONLY);
    if (devnull >= 0) dup2(devnull, 0);
    dup2(fds[1], 1);
    dup2(fds[1], 2);
    execvp(cargv[0], cargv.data());
    const int e = errno;
    (void)!write(err_fds[1], &e, sizeof e);
    _exit(127);
  }
  close(fds[1]);
  close(err_fds[1]);
  if (pid < 0) {
    close(fds[0]);
    close(err_fds[0]);
    result.error = std::string("fork: ") + std::strerror(errno);
    return result;
  }
  int exec_errno = 0;
  const ssize_t got = read(err_fds[0], &exec_errno, sizeof exec_errno);
  close(err_fds[0]);
  if (got == static_cast<ssize_t>(sizeof exec_errno)) {
    close(fds[0]);
    waitpid(pid, nullptr, 0);
    result.error =
        "cannot run '" + argv[0] + "': " + std::strerror(exec_errno);
    return result;
  }
  result.started = true;

  std::string pending;
  char buf[4096];
  bool reading = true;
  bool term_sent = false;
  std::chrono::steady_clock::time_point term_time;
  while (reading) {
    if (cancel && cancel->load()) {
      const auto now = std::chrono::steady_clock::now();
      if (!term_sent) {
        kill(-pid, SIGTERM);
        term_sent = true;
        term_time = now;
        result.cancelled = true;
      } else if (now - term_time > std::chrono::seconds(10)) {
        kill(-pid, SIGKILL);
      }
    }
    pollfd p{fds[0], POLLIN, 0};
    const int pr = poll(&p, 1, 200);
    if (pr < 0 && errno != EINTR) break;
    if (pr <= 0) continue;
    const ssize_t n = read(fds[0], buf, sizeof buf);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) {
      reading = false;
      break;
    }
    pending.append(buf, static_cast<size_t>(n));
    size_t start = 0;
    for (size_t nl; (nl = pending.find_first_of("\r\n", start)) !=
                    std::string::npos;
         start = nl + 1) {
      if (nl > start && on_line) on_line(pending.substr(start, nl - start));
    }
    pending.erase(0, start);
  }
  if (!pending.empty() && on_line) on_line(pending);
  close(fds[0]);

  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result.term_signal = WTERMSIG(status);
  }
  // Children left in the group (a crashed decoder's workers) are not ours to
  // keep running.
  if (term_sent || result.exit_code != 0) kill(-pid, SIGKILL);
  return result;
}

}  // namespace domesday
