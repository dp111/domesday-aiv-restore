/*
 * File:        domesday_model_decoder_stage.h
 * Module:      orc-plugin_domesday-model-decoder
 * Purpose:     Sink stage that decodes PAL video with the Domesday model
 *              decoder (newdecode) by handing the frames to it as an
 *              ld-decode .tbc + .tbc.json
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <orc/plugin/orc_stage_runtime.h>
#include <orc/stage/node_type.h>
#include <orc/stage/params/stage_parameter.h>
#include <orc/stage/stage.h>
#include <orc/stage/triggerable_stage.h>
#include <orc/stage/video_frame_representation.h>
#include <orc/support/stage_instructions.h>

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace domesday {

struct DecoderSettings {
  std::string output_path;     // .y4m written by newdecode
  std::string python = "python3";
  std::string newdecode_path;  // .../newdecode/newdecode.py
  std::string wrapper_path;    // optional launcher, e.g. nd_gpuwrap.py
  std::string scratch_dir;     // where the work directory goes
  std::string extra_args;
  bool video = true;
  bool dropout_mask = false;
  bool keep_work = false;
};

class DomesdayModelDecoderStage : public orc::DAGStage,
                                  public orc::ParameterizedStage,
                                  public orc::TriggerableStage {
 public:
  DomesdayModelDecoderStage();

  // DAGStage
  std::string version() const override { return "0.1.0"; }
  ORC_STAGE_INSTRUCTIONS_MD
  orc::NodeTypeInfo get_node_type_info() const override;
  std::vector<orc::ArtifactPtr> execute(
      const std::vector<orc::ArtifactPtr>& inputs,
      const std::map<std::string, orc::ParameterValue>& parameters,
      orc::ObservationContext& observation_context) override;
  size_t required_input_count() const override { return 1; }
  size_t output_count() const override { return 0; }

  // ParameterizedStage
  std::vector<orc::ParameterDescriptor> get_parameter_descriptors(
      orc::VideoSystem project_format,
      orc::SourceType source_type) const override;
  using orc::ParameterizedStage::get_parameter_descriptors;
  std::map<std::string, orc::ParameterValue> get_parameters() const override;
  bool set_parameters(
      const std::map<std::string, orc::ParameterValue>& params) override;

  // TriggerableStage
  bool trigger(const std::vector<orc::ArtifactPtr>& inputs,
               const std::map<std::string, orc::ParameterValue>& parameters,
               orc::IObservationContext& observation_context) override;
  std::string get_trigger_status() const override { return trigger_status_; }
  void set_progress_callback(orc::TriggerProgressCallback callback) override {
    progress_callback_ = std::move(callback);
  }
  bool is_trigger_in_progress() const override { return processing_.load(); }
  void cancel_trigger() override { cancel_requested_.store(true); }

  // The command line trigger() runs (exposed for tests).
  static std::vector<std::string> build_command(const DecoderSettings& s,
                                                const std::string& tbc_path,
                                                const std::string& scratch);

  // Writes |vfr| as <dir>/in.tbc, in.tbc.json and in.held.json.  Returns an
  // empty string on success, else the error.  (Exposed for tests.)
  std::string export_tbc(const orc::VideoFrameRepresentation& vfr,
                         const std::string& dir);

 private:
  bool apply(DecoderSettings& s,
             const std::map<std::string, orc::ParameterValue>& params,
             std::string* error) const;
  void progress(size_t done, size_t total, const std::string& what) const;

  DecoderSettings settings_;
  std::string trigger_status_;
  orc::TriggerProgressCallback progress_callback_;
  std::atomic<bool> processing_{false};
  std::atomic<bool> cancel_requested_{false};
};

}  // namespace domesday
