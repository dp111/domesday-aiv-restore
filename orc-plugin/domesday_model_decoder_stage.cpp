/*
 * File:        domesday_model_decoder_stage.cpp
 * Module:      orc-plugin_domesday-model-decoder
 * Purpose:     Sink stage bridging decode-orc to the newdecode PAL decoder
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "domesday_model_decoder_stage.h"

#include <orc/plugin/orc_plugin_services_helpers.h>
#include <orc/stage/error_types.h>
#include <orc/stage/observation/colour_frame_phase_query.h>

#include <algorithm>
#include <deque>
#include <filesystem>
#include <fstream>
#include <variant>

#include "ld_export.h"
#include "subprocess_runner.h"

namespace fs = std::filesystem;

namespace domesday {

namespace {

bool as_bool(const orc::ParameterValue& v, bool* out) {
  if (const bool* b = std::get_if<bool>(&v)) {
    *out = *b;
  } else if (const int32_t* i = std::get_if<int32_t>(&v)) {
    *out = *i != 0;
  } else if (const uint32_t* u = std::get_if<uint32_t>(&v)) {
    *out = *u != 0;
  } else if (const std::string* s = std::get_if<std::string>(&v)) {
    if (*s == "1" || *s == "true" || *s == "yes" || *s == "on") {
      *out = true;
    } else if (*s == "0" || *s == "false" || *s == "no" || *s == "off") {
      *out = false;
    } else {
      return false;
    }
  } else {
    return false;
  }
  return true;
}

orc::ParameterDescriptor path_param(const std::string& name,
                                    const std::string& display,
                                    const std::string& description,
                                    const std::string& def, bool required,
                                    const std::string& ext, bool output) {
  orc::ParameterDescriptor d;
  d.name = name;
  d.display_name = display;
  d.description = description;
  d.type = orc::ParameterType::FILE_PATH;
  d.constraints.default_value = def;
  d.constraints.required = required;
  d.file_extension_hint = ext;
  d.output_path = output;
  return d;
}

orc::ParameterDescriptor bool_param(const std::string& name,
                                    const std::string& display,
                                    const std::string& description, bool def) {
  orc::ParameterDescriptor d;
  d.name = name;
  d.display_name = display;
  d.description = description;
  d.type = orc::ParameterType::BOOL;
  d.constraints.default_value = def;
  return d;
}

}  // namespace

DomesdayModelDecoderStage::DomesdayModelDecoderStage() {
  set_configuration_status(orc::ConfigurationStatus::Red);
}

orc::NodeTypeInfo DomesdayModelDecoderStage::get_node_type_info() const {
  return orc::NodeTypeInfo{
      orc::NodeType::SINK,
      "domesday_model_decoder",
      "Domesday model decoder (newdecode)",
      "Decodes PAL video to a 928x576 yuv444p16 .y4m with the Domesday "
      "model decoder (newdecode, Python). Trigger to decode all frames.",
      1,
      1,
      0,
      0,
      orc::VideoFormatCompatibility::PAL_ONLY};
}

std::vector<orc::ArtifactPtr> DomesdayModelDecoderStage::execute(
    const std::vector<orc::ArtifactPtr>&,
    const std::map<std::string, orc::ParameterValue>&,
    orc::ObservationContext&) {
  return {};  // a sink: the work happens in trigger()
}

std::vector<orc::ParameterDescriptor>
DomesdayModelDecoderStage::get_parameter_descriptors(orc::VideoSystem,
                                                     orc::SourceType) const {
  std::vector<orc::ParameterDescriptor> d;
  d.push_back(path_param(
      "output_path", "Output .y4m",
      "Decoded video: 928x576 yuv444p16 YUV4MPEG2, written by newdecode.", "",
      true, ".y4m", true));
  d.push_back(path_param("python", "Python interpreter",
                         "Python with newdecode's dependencies (numpy, scipy, "
                         "PyTorch with CUDA).",
                         "python3", true, "", false));
  d.push_back(path_param("newdecode_path", "newdecode.py",
                         "Path to newdecode/newdecode.py (domesday-aiv-restore).",
                         "", true, ".py", false));
  d.push_back(path_param(
      "wrapper_path", "Launcher script (optional)",
      "Optional Python launcher run as `python LAUNCHER newdecode.py ARGS`, "
      "e.g. nd_gpuwrap.py. Empty: run newdecode.py directly.",
      "", false, ".py", false));
  d.push_back(path_param(
      "scratch_dir", "Scratch directory",
      "Where the work directory (<output>.ndwork: exported .tbc/.tbc.json, "
      "newdecode scratch, de-smeared cache, log) is created. Needs about 3x "
      "the TBC size (1.4 MB/frame each). Empty: beside the output.",
      "", false, "", false));
  d.push_back(bool_param("video", "Solve interlaced video",
                         "newdecode --video 1: solve interlaced frames with the "
                         "model (else they stay transform3d).",
                         true));
  d.push_back(bool_param(
      "dropout_mask", "Dropout mask",
      "newdecode --dropout_mask 1: give the input's dropouts zero data weight "
      "so the solver fills them (for single-copy sources).",
      false));
  {
    orc::ParameterDescriptor e;
    e.name = "extra_args";
    e.display_name = "Extra arguments";
    e.description =
        "Extra newdecode arguments, shell-style quoting (e.g. --luma_lpf 0).";
    e.type = orc::ParameterType::STRING;
    e.constraints.default_value = std::string();
    d.push_back(e);
  }
  d.push_back(bool_param("keep_work", "Keep work directory",
                         "Keep the exported TBC, metadata and log after a "
                         "successful decode (always kept on failure).",
                         false));
  return d;
}

std::map<std::string, orc::ParameterValue>
DomesdayModelDecoderStage::get_parameters() const {
  return {{"output_path", settings_.output_path},
          {"python", settings_.python},
          {"newdecode_path", settings_.newdecode_path},
          {"wrapper_path", settings_.wrapper_path},
          {"scratch_dir", settings_.scratch_dir},
          {"video", settings_.video},
          {"dropout_mask", settings_.dropout_mask},
          {"extra_args", settings_.extra_args},
          {"keep_work", settings_.keep_work}};
}

bool DomesdayModelDecoderStage::apply(
    DecoderSettings& s,
    const std::map<std::string, orc::ParameterValue>& params,
    std::string* error) const {
  std::map<std::string, std::string*> strings{
      {"output_path", &s.output_path},   {"python", &s.python},
      {"newdecode_path", &s.newdecode_path}, {"wrapper_path", &s.wrapper_path},
      {"scratch_dir", &s.scratch_dir},   {"extra_args", &s.extra_args}};
  std::map<std::string, bool*> bools{{"video", &s.video},
                                     {"dropout_mask", &s.dropout_mask},
                                     {"keep_work", &s.keep_work}};
  for (const auto& [name, value] : params) {
    if (auto it = strings.find(name); it != strings.end()) {
      const std::string* v = std::get_if<std::string>(&value);
      if (!v) {
        if (error) *error = name + " must be a string";
        return false;
      }
      *it->second = *v;
    } else if (auto b = bools.find(name); b != bools.end()) {
      if (!as_bool(value, b->second)) {
        if (error) *error = name + " must be a boolean";
        return false;
      }
    }
    // Unknown names are ignored (forward compatibility with project files).
  }
  return true;
}

bool DomesdayModelDecoderStage::set_parameters(
    const std::map<std::string, orc::ParameterValue>& params) {
  DecoderSettings s = settings_;
  std::string error;
  if (!apply(s, params, &error)) {
    ORC_PLUGIN_LOG_ERROR("domesday_model_decoder: {}", error);
    return false;
  }
  settings_ = s;
  const bool ready = !settings_.output_path.empty() &&
                     !settings_.newdecode_path.empty() &&
                     !settings_.python.empty();
  set_configuration_status(ready ? orc::ConfigurationStatus::Green
                                 : orc::ConfigurationStatus::Red);
  return true;
}

void DomesdayModelDecoderStage::progress(size_t done, size_t total,
                                         const std::string& what) const {
  if (progress_callback_) progress_callback_(done, total, what);
}

std::vector<std::string> DomesdayModelDecoderStage::build_command(
    const DecoderSettings& s, const std::string& tbc_path,
    const std::string& scratch) {
  std::vector<std::string> cmd{s.python};
  if (!s.wrapper_path.empty()) cmd.push_back(s.wrapper_path);
  cmd.insert(cmd.end(), {s.newdecode_path, tbc_path, s.output_path,
                         "--scratch", scratch, "--video", s.video ? "1" : "0",
                         "--dropout_mask", s.dropout_mask ? "1" : "0"});
  for (auto& a : split_args(s.extra_args)) cmd.push_back(std::move(a));
  return cmd;
}

std::string DomesdayModelDecoderStage::export_tbc(
    const orc::VideoFrameRepresentation& vfr, const std::string& dir) {
  const auto params = vfr.get_video_parameters();
  if (!params || params->system != orc::VideoSystem::PAL) {
    return "newdecode decodes PAL only; the input is not a PAL source";
  }
  if (vfr.has_separate_channels()) {
    return "the input is Y/C (separate luma/chroma); newdecode needs "
           "composite";
  }
  const orc::FrameIDRange range = vfr.frame_range();
  const size_t frames = static_cast<size_t>(vfr.frame_count());
  if (frames == 0) return "the input has no frames";

  const std::string tbc_path = dir + "/in.tbc";
  std::ofstream tbc(tbc_path, std::ios::binary | std::ios::trunc);
  if (!tbc) return "cannot write " + tbc_path;

  // Spec levels for a level-normalised orc pipeline: black (= blanking for
  // PAL) and white, widened to 16 bit.  newdecode and ld-chroma-decoder take
  // their luma scaling from these.
  LdVideoParameters vp;
  vp.black_16b_ire = (params->black_level >= 0 ? params->black_level : 256) *
                     kTbcScale;
  vp.white_16b_ire = (params->white_level > 0 ? params->white_level : 844) *
                     kTbcScale;
  vp.is_widescreen = params->is_widescreen;
  vp.git_branch = params->git_branch;
  vp.git_commit = params->git_commit;
  const uint16_t blank16 = static_cast<uint16_t>(
      (params->blanking_level >= 0 ? params->blanking_level : 256) *
      kTbcScale);

  std::vector<LdField> fields;
  std::vector<HeldHint> held;
  std::vector<uint16_t> field_buf(
      static_cast<size_t>(kPalTbcFieldWidth) * kPalTbcFieldHeight);
  size_t index = 0;
  for (orc::FrameID id = range.first; range.contains(id) && index < frames;
       ++id, ++index) {
    if (cancel_requested_.load()) return "cancelled";
    const auto desc = vfr.get_frame_descriptor(id);
    const bool pad = !desc || desc->is_padding_frame;
    LdField f1, f2;
    f1.is_first_field = true;
    f2.is_first_field = false;
    f1.seq_no = static_cast<int>(2 * index + 1);
    f2.seq_no = static_cast<int>(2 * index + 2);
    f1.pad = f2.pad = pad;
    if (!pad) {
      const auto phase = orc::observation::measure_frame_phase(vfr, id);
      f1.field_phase_id = phase.field1_phase_id;
      f2.field_phase_id = phase.field2_phase_id;
      for (const auto& run : vfr.get_dropout_hints(id)) {
        split_pal_dropout_run(run, f1.dropouts, f2.dropouts);
      }
      if (desc->held_checked || desc->held_frames_before ||
          desc->held_frames_after) {
        held.push_back({index, static_cast<int64_t>(id), desc->held_checked,
                        desc->held_frames_before, desc->held_frames_after});
      }
    }
    for (int field = 0; field < 2; ++field) {
      const int first_line = field == 0 ? 0 : kPalField1Lines;
      const int lines = field == 0 ? kPalField1Lines
                                   : kPalFrameLines - kPalField1Lines;
      std::fill(field_buf.begin(), field_buf.end(), blank16);
      for (int l = 0; l < lines && !pad; ++l) {
        const int16_t* src =
            vfr.get_line(id, static_cast<size_t>(first_line + l));
        if (!src) continue;  // missing line: leave blanking
        uint16_t* dst = field_buf.data() + static_cast<size_t>(l) *
                                               kPalTbcFieldWidth;
        for (int x = 0; x < kPalTbcFieldWidth; ++x) {
          dst[x] = static_cast<uint16_t>(
              std::clamp<int32_t>(src[x] * kTbcScale, 0, 65535));
        }
      }
      tbc.write(reinterpret_cast<const char*>(field_buf.data()),
                static_cast<std::streamsize>(field_buf.size() * 2));
    }
    fields.push_back(std::move(f1));
    fields.push_back(std::move(f2));
    if (index % 10 == 0) {
      progress(index, frames,
               "Exporting frame " + std::to_string(index) + "/" +
                   std::to_string(frames));
    }
  }
  tbc.close();
  if (!tbc) return "write failed on " + tbc_path + " (disk full?)";

  vp.number_of_sequential_fields = static_cast<int>(fields.size());
  std::ofstream(dir + "/in.tbc.json") << make_ld_json(vp, fields);
  std::ofstream(dir + "/in.held.json") << make_held_hints_json(held);
  ORC_PLUGIN_LOG_INFO(
      "domesday_model_decoder: exported {} frames ({} with held hints) to {}",
      fields.size() / 2, held.size(), tbc_path);
  return {};
}

bool DomesdayModelDecoderStage::trigger(
    const std::vector<orc::ArtifactPtr>& inputs,
    const std::map<std::string, orc::ParameterValue>& parameters,
    orc::IObservationContext&) {
  processing_.store(true);
  cancel_requested_.store(false);
  const auto fail = [this](const std::string& why) {
    trigger_status_ = "Error: " + why;
    ORC_PLUGIN_LOG_ERROR("domesday_model_decoder: {}", why);
    processing_.store(false);
    return false;
  };

  DecoderSettings s = settings_;
  std::string error;
  if (!apply(s, parameters, &error)) return fail(error);
  if (s.output_path.empty()) return fail("output_path is not set");
  if (s.newdecode_path.empty()) return fail("newdecode_path is not set");
  if (!fs::is_regular_file(s.newdecode_path)) {
    return fail("newdecode not found: " + s.newdecode_path);
  }
  if (!s.wrapper_path.empty() && !fs::is_regular_file(s.wrapper_path)) {
    return fail("launcher not found: " + s.wrapper_path);
  }
  if (inputs.empty()) return fail("no input connected");
  const auto vfr =
      std::dynamic_pointer_cast<const orc::VideoFrameRepresentation>(
          inputs[0]);
  if (!vfr) return fail("input is not a video frame representation");

  // Work directory: newdecode keeps its de-smeared cache in the parent of
  // --scratch, so each output gets a directory of its own.
  const fs::path out(s.output_path);
  const fs::path work =
      (s.scratch_dir.empty() ? fs::absolute(out).parent_path()
                             : fs::path(s.scratch_dir)) /
      (out.filename().string() + ".ndwork");
  std::error_code ec;
  fs::create_directories(work / "scratch", ec);
  if (ec) return fail("cannot create " + work.string() + ": " + ec.message());

  trigger_status_ = "Exporting TBC...";
  if (std::string why = export_tbc(*vfr, work.string()); !why.empty()) {
    return fail(why);
  }

  const auto cmd =
      build_command(s, (work / "in.tbc").string(), (work / "scratch").string());
  std::string cmdline;
  for (const auto& a : cmd) cmdline += (cmdline.empty() ? "" : " ") + a;
  ORC_PLUGIN_LOG_INFO("domesday_model_decoder: running {}", cmdline);
  trigger_status_ = "Decoding...";

  std::ofstream log(work / "newdecode.log");
  log << "$ " << cmdline << "\n";
  std::deque<std::string> tail;
  const size_t frames = vfr->frame_count();
  progress(0, frames, "newdecode: starting");
  const SubprocessResult r = run_subprocess(
      cmd,
      [&](const std::string& line) {
        log << line << "\n";
        log.flush();
        tail.push_back(line);
        if (tail.size() > 20) tail.pop_front();
        if (const auto p = parse_progress_line(line)) {
          progress(p->first, p->second,
                   "newdecode frame " + std::to_string(p->first) + "/" +
                       std::to_string(p->second));
        } else {
          ORC_PLUGIN_LOG_DEBUG("{}", line);
        }
      },
      &cancel_requested_);
  log.close();

  if (!r.started) return fail(r.error);
  if (r.cancelled) return fail("cancelled; work kept in " + work.string());
  if (r.exit_code != 0) {
    std::string last;
    for (const auto& l : tail) last += "\n  " + l;
    ORC_PLUGIN_LOG_ERROR("domesday_model_decoder: newdecode output tail:{}",
                         last);
    return fail("newdecode failed (" +
                (r.term_signal ? "signal " + std::to_string(r.term_signal)
                               : "exit " + std::to_string(r.exit_code)) +
                "); see " + (work / "newdecode.log").string());
  }
  if (!fs::is_regular_file(s.output_path)) {
    return fail("newdecode exited 0 but wrote no " + s.output_path);
  }
  if (!s.keep_work) fs::remove_all(work, ec);
  progress(frames, frames, "newdecode: done");
  trigger_status_ = "Decoded " + std::to_string(frames) + " frames to " +
                    s.output_path;
  processing_.store(false);
  return true;
}

}  // namespace domesday
