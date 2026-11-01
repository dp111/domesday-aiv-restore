/*
 * File:        plugin.cpp
 * Module:      orc-plugin_domesday-model-decoder
 * Purpose:     Plugin bundle for the Domesday model decoder sink
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <orc/abi/orc_plugin_sdk.h>

#include "domesday_model_decoder_stage.h"

#ifndef ORC_STAGE_PLUGIN_VERSION
#define ORC_STAGE_PLUGIN_VERSION "dev"
#endif

namespace {

constexpr orc::StagePluginDescriptor kPluginDescriptor =
    ORC_STAGE_PLUGIN_DESCRIPTOR("io.github.dp111.domesday_model_decoder",
                                ORC_STAGE_PLUGIN_VERSION, "GPL-3.0-or-later",
                                false);

}  // namespace

ORC_DEFINE_STAGE_PLUGIN(kPluginDescriptor,
                        domesday::DomesdayModelDecoderStage)
