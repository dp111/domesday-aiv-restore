/*
 * File:        ld_export_test.cpp
 * Purpose:     Unit tests for the host-independent bridge helpers
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gtest/gtest.h>
#include <orc/support/dropout_util.h>

#include "ld_export.h"

using namespace domesday;

namespace {

orc::DropoutRun run_at(int field, int line, int x, uint32_t count) {
  orc::DropoutRun r;
  r.sample_start = orc::dropout_util::field_line_to_frame_sample(
      orc::VideoSystem::PAL, field, line, x);
  r.sample_count = count;
  return r;
}

}  // namespace

TEST(SplitPalDropoutRun, RoundTripsTbcSourceCoordinates) {
  // What tbc_source builds from fieldLine 47, startx 38, endx 44 on the
  // second field must come back unchanged (fieldLine is 1-based).
  std::vector<LdDropout> f1, f2;
  split_pal_dropout_run(run_at(2, 46, 38, 6), f1, f2);
  ASSERT_TRUE(f1.empty());
  ASSERT_EQ(f2.size(), 1u);
  EXPECT_EQ(f2[0].field_line, 47);
  EXPECT_EQ(f2[0].startx, 38);
  EXPECT_EQ(f2[0].endx, 44);
}

TEST(SplitPalDropoutRun, SplitsAtLineEnd) {
  std::vector<LdDropout> f1, f2;
  split_pal_dropout_run(run_at(1, 100, 1130, 10), f1, f2);
  ASSERT_EQ(f1.size(), 2u);
  EXPECT_EQ(f1[0].field_line, 101);
  EXPECT_EQ(f1[0].startx, 1130);
  EXPECT_EQ(f1[0].endx, 1135);
  EXPECT_EQ(f1[1].field_line, 102);
  EXPECT_EQ(f1[1].startx, 0);
  EXPECT_EQ(f1[1].endx, 5);
}

TEST(SplitPalDropoutRun, ClipsLongPalLineToStoredWidth) {
  // Frame line 312 (field 1, last line) carries 1137 samples; the .tbc keeps
  // 1135, so the extra two are dropped rather than described.
  std::vector<LdDropout> f1, f2;
  split_pal_dropout_run(run_at(1, 312, 1130, 7), f1, f2);
  ASSERT_EQ(f1.size(), 1u);
  EXPECT_EQ(f1[0].endx, 1135);
  EXPECT_TRUE(f2.empty());
}

TEST(MakeLdJson, CarriesWhatNewdecodeAndChromaDecoderRead) {
  LdVideoParameters vp;
  vp.number_of_sequential_fields = 2;
  LdField a;
  a.seq_no = 1;
  a.field_phase_id = 3;
  LdField b;
  b.is_first_field = false;
  b.seq_no = 2;
  b.dropouts = {{47, 38, 44}, {210, 23, 24}};
  const std::string j = make_ld_json(vp, {a, b});
  for (const char* s :
       {"\"system\": \"PAL\"", "\"numberOfSequentialFields\": 2",
        "\"activeVideoStart\": 185", "\"activeVideoEnd\": 1107",
        "\"black16bIre\": 16384", "\"white16bIre\": 54016",
        "\"fieldPhaseID\": 3", "\"isFirstField\": false",
        "\"fieldLine\": [47,210]", "\"startx\": [38,23]",
        "\"endx\": [44,24]"}) {
    EXPECT_NE(j.find(s), std::string::npos) << s;
  }
}

TEST(ParseProgressLine, ReadsNewdecodeFrameLines) {
  auto p = parse_progress_line("[newdecode] frame 8/12  solve 1.2 s");
  ASSERT_TRUE(p.has_value());
  EXPECT_EQ(p->first, 8u);
  EXPECT_EQ(p->second, 12u);
  EXPECT_FALSE(parse_progress_line("[newdecode] frames: 12").has_value());
  EXPECT_FALSE(parse_progress_line("levels from TBC: ...").has_value());
}

TEST(SplitArgs, HonoursQuotes) {
  const auto a = split_args(" --luma_lpf 0  --note 'a b' \"c\\\"d\" ");
  ASSERT_EQ(a.size(), 5u);
  EXPECT_EQ(a[0], "--luma_lpf");
  EXPECT_EQ(a[3], "a b");
  EXPECT_EQ(a[4], "c\"d");
  EXPECT_TRUE(split_args("   ").empty());
}
