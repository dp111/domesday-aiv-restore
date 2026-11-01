/*
 * File:        ld_export.cpp
 * Module:      orc-plugin_domesday-model-decoder
 * Purpose:     Host-independent helpers for the newdecode bridge
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "ld_export.h"

#include <orc/support/dropout_util.h>
#include <orc/support/frame_line_util.h>

#include <algorithm>
#include <cctype>
#include <sstream>

namespace domesday {

void split_pal_dropout_run(const orc::DropoutRun& run,
                           std::vector<LdDropout>& field1,
                           std::vector<LdDropout>& field2) {
  const uint64_t end = run.sample_start + run.sample_count;
  uint64_t offset = run.sample_start;
  while (offset < end) {
    const auto fls = orc::dropout_util::frame_sample_to_field_line(
        orc::VideoSystem::PAL, offset);
    const int frame_line =
        fls.field == 1 ? fls.line : kPalField1Lines + fls.line;
    const auto line_width = static_cast<int64_t>(orc::frame_line_sample_count(
        orc::VideoSystem::PAL, kPalTbcFieldWidth,
        static_cast<size_t>(frame_line)));
    const uint64_t on_line = std::min<uint64_t>(
        static_cast<uint64_t>(line_width - fls.sample), end - offset);
    const int x0 = fls.sample;
    const int x1 = std::min(fls.sample + static_cast<int>(on_line),
                            kPalTbcFieldWidth);
    if (x1 > x0) {
      (fls.field == 1 ? field1 : field2).push_back({fls.line + 1, x0, x1});
    }
    offset += on_line;
  }
}

namespace {

std::string json_string(const std::string& s) {
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof buf, "\\u%04x", c);
      out += buf;
    } else {
      out += c;
    }
  }
  return out + "\"";
}

const char* json_bool(bool b) { return b ? "true" : "false"; }

template <typename F>
void json_int_array(std::ostringstream& o, const std::vector<LdDropout>& d,
                    F member) {
  o << '[';
  for (size_t i = 0; i < d.size(); ++i) o << (i ? "," : "") << member(d[i]);
  o << ']';
}

}  // namespace

std::string make_ld_json(const LdVideoParameters& vp,
                         const std::vector<LdField>& fields) {
  std::ostringstream o;
  o << "{\"videoParameters\": {"
    << "\"system\": \"PAL\", \"isSourcePal\": true"
    << ", \"numberOfSequentialFields\": " << vp.number_of_sequential_fields
    << ", \"sampleRate\": 17734475.0"
    << ", \"fieldWidth\": " << kPalTbcFieldWidth
    << ", \"fieldHeight\": " << kPalTbcFieldHeight
    << ", \"activeVideoStart\": " << vp.active_video_start
    << ", \"activeVideoEnd\": " << vp.active_video_end
    << ", \"colourBurstStart\": " << vp.colour_burst_start
    << ", \"colourBurstEnd\": " << vp.colour_burst_end
    << ", \"white16bIre\": " << vp.white_16b_ire
    << ", \"black16bIre\": " << vp.black_16b_ire
    << ", \"isMapped\": false, \"isSubcarrierLocked\": false"
    << ", \"isWidescreen\": " << json_bool(vp.is_widescreen)
    << ", \"gitBranch\": " << json_string(vp.git_branch)
    << ", \"gitCommit\": " << json_string(vp.git_commit) << "},\n\"fields\": [";
  for (size_t i = 0; i < fields.size(); ++i) {
    const LdField& f = fields[i];
    o << (i ? ",\n" : "\n") << "{\"isFirstField\": "
      << json_bool(f.is_first_field) << ", \"seqNo\": " << f.seq_no;
    if (f.field_phase_id >= 0) o << ", \"fieldPhaseID\": " << f.field_phase_id;
    if (f.pad) o << ", \"pad\": true";
    if (!f.dropouts.empty()) {
      o << ", \"dropOuts\": {\"fieldLine\": ";
      json_int_array(o, f.dropouts, [](const LdDropout& d) { return d.field_line; });
      o << ", \"startx\": ";
      json_int_array(o, f.dropouts, [](const LdDropout& d) { return d.startx; });
      o << ", \"endx\": ";
      json_int_array(o, f.dropouts, [](const LdDropout& d) { return d.endx; });
      o << '}';
    }
    o << '}';
  }
  o << "\n]}\n";
  return o.str();
}

std::string make_held_hints_json(const std::vector<HeldHint>& hints) {
  std::ostringstream o;
  o << "{\"format\": \"domesday-held-hints-1\", \"frames\": [";
  for (size_t i = 0; i < hints.size(); ++i) {
    const HeldHint& h = hints[i];
    o << (i ? ",\n" : "\n") << "{\"index\": " << h.index
      << ", \"frameId\": " << h.frame_id
      << ", \"heldChecked\": " << json_bool(h.held_checked)
      << ", \"heldFramesBefore\": " << h.held_frames_before
      << ", \"heldFramesAfter\": " << h.held_frames_after << '}';
  }
  o << "\n]}\n";
  return o.str();
}

std::optional<std::pair<uint64_t, uint64_t>> parse_progress_line(
    const std::string& line) {
  static const std::string kPrefix = "[newdecode] frame ";
  if (line.compare(0, kPrefix.size(), kPrefix) != 0) return std::nullopt;
  unsigned long long done = 0;
  unsigned long long total = 0;
  char slash = 0;
  std::istringstream in(line.substr(kPrefix.size()));
  if (!(in >> done) || !(in.get(slash)) || slash != '/' || !(in >> total) ||
      total == 0) {
    return std::nullopt;
  }
  return std::make_pair(static_cast<uint64_t>(done),
                        static_cast<uint64_t>(total));
}

std::vector<std::string> split_args(const std::string& text) {
  std::vector<std::string> out;
  std::string cur;
  bool in_word = false;
  char quote = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (quote) {
      if (c == quote) {
        quote = 0;
      } else if (c == '\\' && quote == '"' && i + 1 < text.size()) {
        cur += text[++i];
      } else {
        cur += c;
      }
    } else if (c == '\'' || c == '"') {
      quote = c;
      in_word = true;
    } else if (c == '\\' && i + 1 < text.size()) {
      cur += text[++i];
      in_word = true;
    } else if (std::isspace(static_cast<unsigned char>(c))) {
      if (in_word) out.push_back(cur);
      cur.clear();
      in_word = false;
    } else {
      cur += c;
      in_word = true;
    }
  }
  if (in_word) out.push_back(cur);
  return out;
}

}  // namespace domesday
