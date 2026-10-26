# Machine settings for every script in this repo.  Sourced, never run.
# Put your own values in config.local.sh (not tracked; see config.local.example.sh) or in the environment.
DAR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)          # this repo
[ -f "$DAR/config.local.sh" ] && . "$DAR/config.local.sh"

: "${PY:=python3}"                  # Python with ld-decode's requirements (numpy, scipy, numba, av)
: "${MLPY:=$PY}"                    # Python with PyTorch + CUDA, for newdecode
: "${LD:=$HOME/ld-decode}"          # github.com/dp111/ld-decode, branch domesday-aiv (decoder + stacker)
: "${WORK:=$HOME/aiv-work}"         # pipeline state, logs and capture records
: "${V7_OUT:=$WORK/out}"            # stacks, masters (<side>.v8.mkv), logs per side
: "${V7_SCRROOT:=$WORK/scratch}"    # per-side scratch (dropout-corrected stack, newdecode chunks)
: "${V7_SLOTROOT:=$WORK/pvf}"       # PVF install root: <root>/BeebVFS<slot>/video.pvf (Pi1MHz layout)
: "${V7_OLDF:=}"                    # optional: an earlier set of stacks to gate new ones against
# capture decodes: "dir:mount:reserve_GB ..." - each decode goes to the mount with most room above its reserve
: "${V7_DRIVES:=$WORK/decodes:$WORK:100}"
# capture roots used by pipeline/sides.conf
: "${CAP_AIV:=$WORK/captures/domesday}"   # Domesday_DD86-DS*_<side>_*.ldf, one directory per disc set
: "${CAP_DISCS:=$WORK/captures}"          # the other titles (EcoDisc, Countryside, City_disc, ...)
: "${CAP_UN:=$WORK/captures/UN}"          # UN_Disc[1-7]_CAV_PAL.raw.oga
export PY MLPY LD WORK V7_OUT V7_SCRROOT V7_SLOTROOT V7_OLDF
