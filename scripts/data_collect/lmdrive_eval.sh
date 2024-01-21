#!/bin/bash
export SETTING_DIR=$(dirname "$(realpath -s "$BASH_SOURCE")")
export ROOT="$(dirname "$(dirname "$SETTING_DIR")")"
. ${ROOT}/settings.sh

[[ -z "$DATA_ROOT" ]] && export DATA_ROOT=${DATA_ROOT_BASE}/langprop/lmdrive_eval

export TIMEOUT=200000

export TEAM_AGENT=${ROOT}/src/lmdrive/lm_agent_eval.py
export TEAM_CONFIG=${ROOT}/src/lmdrive/config/lmdrive_eval.yaml

bash ${ROOT}/scripts/data_collect/base.sh $@
