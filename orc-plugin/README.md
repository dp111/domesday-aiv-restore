# decode-orc plugin: Domesday model decoder (newdecode) sink

An external [decode-orc](https://github.com/decode-orc/decode-orc) stage plugin
that lets a decode-orc pipeline decode PAL video with this repository's model
decoder, `newdecode/newdecode.py`.

This is a **bridge** (v1): the sink stage `domesday_model_decoder` exports the
pipeline's frames as an ld-decode `.tbc` + `.tbc.json`, runs newdecode on them
as a subprocess, and reports its progress back to decode-orc. See
[instructions.md](instructions.md) for what the stage does and its parameters
(the same text is shown by `orc-cli stages help domesday_model_decoder`).

## Build

Needs a decode-orc SDK install (host ABI 19, decode-orc main a690be72 or
later), CMake 3.20+, a C++17 compiler, and optionally GTest for the unit tests.
Linux/POSIX only.

```bash
# 1. decode-orc and its SDK (see decode-orc BUILD.md for dependencies)
cmake -S decode-orc -B decode-orc/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_GUI=OFF -DEZPWD_INCLUDE_DIR=/path/to/ezpwd-reed-solomon/c++
cmake --build decode-orc/build
cmake --install decode-orc/build --prefix $HOME/orc-prefix
#    (only the SDK is needed to *build* the plugin:
#     cmake --build decode-orc/build --target orc-sdk-support &&
#     cmake -DCMAKE_INSTALL_PREFIX=$HOME/orc-prefix -P decode-orc/build/orc/sdk/cmake_install.cmake)

# 2. the plugin
cmake -S orc-plugin -B orc-plugin-build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=$HOME/orc-prefix
cmake --build orc-plugin-build
ctest --test-dir orc-plugin-build          # unit tests (GTest)
```

The plugin is `orc-plugin-build/lib/orc-stage-plugins/liborc-plugin_domesday-model-decoder.so`
with its `instructions.md` copied beside it.

## Install / load

Either point the host at the build directory for one run:

```bash
export ORC_STAGE_PLUGIN_PATHS=$PWD/orc-plugin-build/lib/orc-stage-plugins
orc-cli stages | grep domesday
```

or register it permanently (GUI Plugin Manager or):

```bash
orc-cli plugins add orc-plugin-build/lib/orc-stage-plugins/liborc-plugin_domesday-model-decoder.so --yes
```

## Use

Filtergraph mode (headless):

```bash
orc-cli --source "tbc_source=input_path=disc.tbc" \
  --sink "domesday_model_decoder=output_path=disc.y4m:python=/root/mlenv/bin/python:newdecode_path=/path/to/newdecode/newdecode.py:wrapper_path=/path/to/nd_gpuwrap.py:scratch_dir=/big/disk:video=true"
```

or add the **Domesday model decoder (newdecode)** sink to a project in the GUI
(`orc-gui`) / a `.orcprj`, set its parameters and trigger it
(`orc-cli project.orcprj --process`). Output is 928x576 `yuv444p16` YUV4MPEG2;
encode it with ffmpeg as you would newdecode's own output.

newdecode itself needs Python with numpy, scipy and CUDA PyTorch, and
`ld-chroma-decoder` on `PATH`.

## Notes and limitations

- PAL composite only (the stage is `PAL_ONLY`; Y/C input is refused).
- decode-orc carries video as 10-bit CVBS normalised to spec levels, so a
  TBC that goes through orc is re-quantised (16-bit step 64) and re-levelled
  (the TBC's measured black, e.g. 16067, is mapped to 16384). The exported
  metadata declares those spec levels, so the decode is correct, but it is not
  bit-identical to running newdecode directly on the original 16-bit TBC.
- `activeVideoStart/End` are written as ld-decode's PAL 185/1107 rather than
  orc's canonical 157/1105: ld-chroma-decoder sizes its output from them and
  newdecode expects 928 samples.
- `fieldPhaseID` is measured with orc's colour_frame_phase observer (not used by
  newdecode or ld-chroma-decoder for PAL, but written for other tools).
- Held-picture hints (FrameDescriptor `held_frames_before/after`,
  `held_checked`) are written to `in.held.json` for a future native held
  solve; newdecode does its own held detection today.
- The whole input is exported before the decode starts; work space is about
  3 x 1.42 MB per frame.
