#!/usr/bin/env bash
set -eo pipefail
source "$ASCEND_HOME_PATH/set_env.sh"
source "$ASCEND_HOME_PATH/share/info/ascendnpu-ir/bin/set_env.sh"
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi="${ATB_CXX_ABI:-1}"
source "$ASCEND_HOME_PATH/opp/vendors/customize/bin/set_env.bash"
source "$ASCEND_HOME_PATH/opp/vendors/custom_transformer/bin/set_env.bash"
exec "$@"
