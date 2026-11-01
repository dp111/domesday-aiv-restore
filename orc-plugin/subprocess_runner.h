/*
 * File:        subprocess_runner.h
 * Module:      orc-plugin_domesday-model-decoder
 * Purpose:     Run a child process, streaming its merged stdout/stderr lines
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace domesday {

struct SubprocessResult {
  bool started = false;    // false: the program could not be launched
  bool cancelled = false;  // the cancel flag was raised and the child killed
  int exit_code = -1;      // exit status, or -1 when killed by a signal
  int term_signal = 0;     // the signal that ended it, if any
  std::string error;       // launch error text
};

// Runs argv[0] (searched on PATH) in its own process group with stdout and
// stderr merged into one pipe; on_line receives each line as it arrives.
// When *cancel becomes true the whole process group gets SIGTERM, then
// SIGKILL after a grace period.
SubprocessResult run_subprocess(
    const std::vector<std::string>& argv,
    const std::function<void(const std::string&)>& on_line,
    const std::atomic<bool>* cancel);

}  // namespace domesday
