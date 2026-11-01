/*
 * File:        ld_export.h
 * Module:      orc-plugin_domesday-model-decoder
 * Purpose:     Host-independent helpers for the newdecode bridge: ld-decode
 *              .tbc.json metadata, dropout splitting, held-hint sidecar,
 *              progress-line parsing and argument splitting.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <orc/stage/common_types.h>
#include <orc/stage/dropout/dropout_run.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace domesday {

// PAL geometry of an ld-decode .tbc: two fields per frame, each stored as
// 313 lines of 1135 samples (the 312-line second field gets one padding line).
constexpr int kPalTbcFieldWidth = 1135;
constexpr int kPalTbcFieldHeight = 313;
constexpr int kPalField1Lines = 313;
constexpr int kPalFrameLines = 625;
// CVBS_U10_4FSC -> ld-decode 16-bit: a lossless x64 widening (PAL blanking
// 256 -> 16384, white 844 -> 54016), the same map the core tbc_sink uses.
constexpr int kTbcScale = 64;

// One ld-decode dropout: 1-based field line, [startx, endx) samples.
struct LdDropout {
  int field_line;
  int startx;
  int endx;
};

struct LdField {
  bool is_first_field = true;
  int seq_no = 1;          // 1-based
  int field_phase_id = -1;  // -1: unknown (omitted)
  bool pad = false;
  std::vector<LdDropout> dropouts;
};

// The videoParameters block.  activeVideoStart/End and colourBurstStart/End
// are ld-decode's PAL values (not orc's canonical 157/1105): ld-chroma-decoder
// sizes its output from them, and newdecode expects its 928-sample picture.
struct LdVideoParameters {
  int number_of_sequential_fields = 0;
  int black_16b_ire = 16384;
  int white_16b_ire = 54016;
  bool is_widescreen = false;
  int active_video_start = 185;
  int active_video_end = 1107;
  int colour_burst_start = 98;
  int colour_burst_end = 138;
  std::string git_branch;
  std::string git_commit;
};

// Split one frame-flat DropoutRun of a PAL frame into per-field ld-decode
// dropouts (field 1 = frame lines 0..312, field 2 = 313..624).  Samples beyond
// the stored 1135 (the long PAL lines) are dropped, as the .tbc does not keep
// them.
void split_pal_dropout_run(const orc::DropoutRun& run,
                           std::vector<LdDropout>& field1,
                           std::vector<LdDropout>& field2);

// The complete .tbc.json document (PAL).
std::string make_ld_json(const LdVideoParameters& vp,
                         const std::vector<LdField>& fields);

// Held-picture hints from the FrameDescriptors (ABI 19), for a later held
// solve.  `index` is the 0-based frame index in the exported .tbc.
struct HeldHint {
  uint64_t index = 0;
  int64_t frame_id = 0;
  bool held_checked = false;
  uint32_t held_frames_before = 0;
  uint32_t held_frames_after = 0;
};
std::string make_held_hints_json(const std::vector<HeldHint>& hints);

// "[newdecode] frame 12/40  ..." -> {12, 40}; anything else -> nullopt.
std::optional<std::pair<uint64_t, uint64_t>> parse_progress_line(
    const std::string& line);

// Whitespace-separated arguments; '…' and "…" group, backslash escapes.
std::vector<std::string> split_args(const std::string& text);

}  // namespace domesday
