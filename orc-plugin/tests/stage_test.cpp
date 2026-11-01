/*
 * File:        stage_test.cpp
 * Purpose:     Stage tests: TBC export from a synthetic representation, and
 *              trigger() against a stand-in decoder script
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gtest/gtest.h>
#include <orc/stage/artifact.h>
#include <orc/stage/cvbs_signal_constants.h>
#include <orc/stage/observation/colour_frame_phase_query.h>
#include <orc/support/dropout_util.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "domesday_model_decoder_stage.h"
#include "ld_export.h"

namespace fs = std::filesystem;
using namespace domesday;

namespace {

// Two PAL frames; sample value encodes the frame line so the field split can
// be checked.  Frame 1 carries one dropout and a held hint.
class FakePal : public orc::Artifact, public orc::VideoFrameRepresentation {
 public:
  FakePal() : orc::Artifact(orc::ArtifactID("fake"), orc::Provenance{}) {
    frame_.resize(orc::kPalFrameSamples);
    for (size_t l = 0; l < 625; ++l) {
      const size_t off =
          orc::frame_line_sample_offset(orc::VideoSystem::PAL, 1135, l);
      const size_t n =
          orc::frame_line_sample_count(orc::VideoSystem::PAL, 1135, l);
      for (size_t x = 0; x < n; ++x) frame_[off + x] = int16_t(256 + l % 500);
    }
  }
  std::string type_name() const override { return "FakePal"; }
  orc::FrameIDRange frame_range() const override { return {10, 11}; }
  size_t frame_count() const override { return 2; }
  bool has_frame(orc::FrameID id) const override { return id == 10 || id == 11; }
  std::optional<orc::FrameDescriptor> get_frame_descriptor(
      orc::FrameID id) const override {
    if (!has_frame(id)) return std::nullopt;
    orc::FrameDescriptor d;
    d.frame_id = id;
    d.system = orc::VideoSystem::PAL;
    if (id == 11) {
      d.held_checked = true;
      d.held_frames_before = 1;
    }
    return d;
  }
  const sample_type* get_frame(orc::FrameID id) const override {
    return has_frame(id) ? frame_.data() : nullptr;
  }
  std::vector<sample_type> get_frame_copy(orc::FrameID) const override {
    return frame_;
  }
  std::vector<orc::DropoutRun> get_dropout_hints(
      orc::FrameID id) const override {
    if (id != 11) return {};
    orc::DropoutRun r;
    r.frame_id = id;
    r.sample_start = orc::dropout_util::field_line_to_frame_sample(
        orc::VideoSystem::PAL, 2, 46, 38);
    r.sample_count = 6;
    return {r};
  }
  std::optional<orc::SourceParameters> get_video_parameters() const override {
    orc::SourceParameters p;
    p.system = orc::VideoSystem::PAL;
    p.frame_width_nominal = 1135;
    p.frame_height = 625;
    p.blanking_level = orc::kPalBlanking;
    p.black_level = orc::kPalBlack;
    p.white_level = orc::kPalWhite;
    return p;
  }

 private:
  std::vector<sample_type> frame_;
};

std::string slurp(const fs::path& p) {
  std::ifstream in(p);
  std::stringstream s;
  s << in.rdbuf();
  return s.str();
}

fs::path temp_dir(const std::string& name) {
  const fs::path d = fs::temp_directory_path() / ("domesday_plugin_" + name);
  fs::remove_all(d);
  fs::create_directories(d);
  return d;
}

}  // namespace

TEST(DomesdayStage, ExportsLdDecodeLayout) {
  const fs::path dir = temp_dir("export");
  DomesdayModelDecoderStage stage;
  FakePal vfr;
  ASSERT_EQ(stage.export_tbc(vfr, dir.string()), "");

  const size_t field_samples = 1135 * 313;
  ASSERT_EQ(fs::file_size(dir / "in.tbc"), 2 * 2 * field_samples * 2);
  std::vector<uint16_t> tbc(2 * 2 * field_samples);
  std::ifstream(dir / "in.tbc", std::ios::binary)
      .read(reinterpret_cast<char*>(tbc.data()), tbc.size() * 2);
  EXPECT_EQ(tbc[5 * 1135 + 7], (256 + 5) * 64);  // field 1 line 5
  EXPECT_EQ(tbc[field_samples + 5 * 1135], (256 + 318) * 64);  // frame line 318
  EXPECT_EQ(tbc[2 * field_samples - 1], 256 * 64);  // field 2 padding line

  const std::string j = slurp(dir / "in.tbc.json");
  EXPECT_NE(j.find("\"numberOfSequentialFields\": 4"), std::string::npos);
  EXPECT_NE(j.find("\"seqNo\": 4, \"dropOuts\": {\"fieldLine\": [47], "
                   "\"startx\": [38], \"endx\": [44]}"),
            std::string::npos);
  const std::string h = slurp(dir / "in.held.json");
  EXPECT_NE(h.find("\"index\": 1, \"frameId\": 11, \"heldChecked\": true, "
                   "\"heldFramesBefore\": 1"),
            std::string::npos);
  fs::remove_all(dir);
}

TEST(DomesdayStage, BuildsNewdecodeCommand) {
  DecoderSettings s;
  s.python = "/py";
  s.wrapper_path = "/wrap.py";
  s.newdecode_path = "/nd.py";
  s.output_path = "/o.y4m";
  s.dropout_mask = true;
  s.extra_args = "--luma_lpf 0";
  const std::vector<std::string> want{
      "/py", "/wrap.py", "/nd.py", "/w/in.tbc", "/o.y4m", "--scratch",
      "/w/scratch", "--video", "1", "--dropout_mask", "1", "--luma_lpf", "0"};
  EXPECT_EQ(DomesdayModelDecoderStage::build_command(s, "/w/in.tbc",
                                                     "/w/scratch"),
            want);
}

TEST(DomesdayStage, TriggerRunsDecoderAndReportsProgressAndErrors) {
  const fs::path dir = temp_dir("trigger");
  // Stand-in for newdecode: checks its inputs, prints progress, writes out.
  const fs::path script = dir / "fake_newdecode.sh";
  std::ofstream(script)
      << "test -s \"$1\" && test -s \"$1.json\" || exit 2\n"
         "[ \"$FAIL\" = 1 ] && { echo boom >&2; exit 3; }\n"
         "echo '[newdecode] frame 1/2  x' >&2\n"
         "echo '[newdecode] frame 2/2  x' >&2\n"
         "echo YUV4MPEG2 > \"$2\"\n";
  DomesdayModelDecoderStage stage;
  ASSERT_TRUE(stage.set_parameters(
      {{"output_path", (dir / "out.y4m").string()},
       {"python", std::string("/bin/sh")},
       {"newdecode_path", script.string()}}));
  EXPECT_EQ(stage.get_configuration_status(), orc::ConfigurationStatus::Green);
  std::vector<std::pair<size_t, size_t>> seen;
  stage.set_progress_callback(
      [&](size_t d, size_t t, const std::string& m) {
        if (m.rfind("newdecode frame", 0) == 0) seen.emplace_back(d, t);
      });
  auto vfr = std::make_shared<FakePal>();
  orc::observation::detail::CaptureContext ctx;  // any IObservationContext
  ASSERT_TRUE(stage.trigger({vfr}, stage.get_parameters(), ctx))
      << stage.get_trigger_status();
  EXPECT_TRUE(fs::exists(dir / "out.y4m"));
  EXPECT_FALSE(fs::exists(dir / "out.y4m.ndwork"));  // removed on success
  EXPECT_EQ(seen, (std::vector<std::pair<size_t, size_t>>{{1, 2}, {2, 2}}));

  setenv("FAIL", "1", 1);
  EXPECT_FALSE(stage.trigger({vfr}, stage.get_parameters(), ctx));
  unsetenv("FAIL");
  EXPECT_NE(stage.get_trigger_status().find("exit 3"), std::string::npos)
      << stage.get_trigger_status();
  EXPECT_NE(slurp(dir / "out.y4m.ndwork" / "newdecode.log").find("boom"),
            std::string::npos);

  // An interpreter that cannot be run is reported, not a crash.
  auto params = stage.get_parameters();
  params["python"] = std::string("/nonexistent/python");
  EXPECT_FALSE(stage.trigger({vfr}, params, ctx));
  EXPECT_NE(stage.get_trigger_status().find("cannot run"), std::string::npos)
      << stage.get_trigger_status();
  fs::remove_all(dir);
}
