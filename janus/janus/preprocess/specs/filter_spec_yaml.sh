#!/usr/bin/env bash

# Root directory where the YAML spec folders are located
ROOT_DIR=$(PYTHONPATH="$(dirname "$0")/../.." python - <<'PY'
from janus.utils.paths import project_root
print(project_root())
PY
)
ROOT_DIR="$ROOT_DIR/data/raw_data/3gpp_openapi_yaml_rel17"

allowed_files=(
  ["29502-hd0"]="TS29502_Nsmf_PDUSession.yaml"
  ["29503-hj0"]="TS29503_Nudm_EE.yaml TS29503_Nudm_NIDDAU.yaml TS29503_Nudm_RSDS.yaml TS29503_Nudm_SSAU.yaml TS29503_Nudm_UECM.yaml TS29503_Nudm_MT.yaml TS29503_Nudm_PP.yaml TS29503_Nudm_SDM.yaml TS29503_Nudm_UEAU.yaml TS29503_Nudm_UEID.yaml"
  ["29504-hi0"]="TS29504_Nudr_DR.yaml TS29504_Nudr_GroupIDmap.yaml"
  ["29505-hd0"]="TS29505_Subscription_Data.yaml"
  ["29507-hb0"]="TS29507_Npcf_AMPolicyControl.yaml"
  ["29508-hg0"]="TS29508_Nsmf_EventExposure.yaml"
  ["29509-hb0"]="TS29509_Nausf_SoRProtection.yaml TS29509_Nausf_UEAuthentication.yaml TS29509_Nausf_UPUProtection.yaml"
  ["29510-hg0"]="TS29510_Nnrf_AccessToken.yaml TS29510_Nnrf_NFDiscovery.yaml TS29510_Nnrf_Bootstrapping.yaml TS29510_Nnrf_NFManagement.yaml"
  ["29512-hf0"]="TS29512_Npcf_SMPolicyControl.yaml"
  ["29514-hd0"]="TS29514_Npcf_PolicyAuthorization.yaml"
  ["29517-hb0"]="TS29517_Naf_EventExposure.yaml"
  ["29518-hg0"]="TS29518_Namf_Communication.yaml TS29518_Namf_Location.yaml TS29518_Namf_MBSCommunication.yaml TS29518_Namf_EventExposure.yaml TS29518_Namf_MBSBroadcast.yaml TS29518_Namf_MT.yaml"
  ["29521-ha0"]="TS29521_Nbsf_Management.yaml"
  ["29523-h90"]="TS29523_Npcf_EventExposure.yaml"
  ["29525-ha0"]="TS29525_Npcf_UEPolicyControl.yaml"
  ["29531-hb0"]="TS29531_Nnssf_NSSAIAvailability.yaml TS29531_Nnssf_NSSelection.yaml"
  ["29534-h30"]="TS29534_Npcf_AMPolicyAuthorization.yaml"
  ["29537-h30"]="TS29537_Npcf_MBSPolicyAuthorization.yaml TS29537_Npcf_MBSPolicyControl.yaml"
  ["29542-h50"]="TS29542_Nsmf_NIDD.yaml"
  ["29554-h40"]="TS29554_Npcf_BDTPolicyControl.yaml"
  ["29571-hc0"]="TS29571_CommonData.yaml"
)

# List of directories to keep (whitelisted)
keep_dirs=(
  "29502-hd0"
  "29503-hj0"
  "29504-hi0"
  "29505-hd0"
  "29507-hb0"
  "29508-hg0"
  "29509-hb0"
  "29510-hg0"
  "29512-hf0"
  "29514-hd0"
  "29517-hb0"
  "29518-hg0"
  "29521-ha0"
  "29523-h90"
  "29525-ha0"
  "29531-hb0"
  "29534-h30"
  "29537-h30"
  "29542-h50"
  "29554-h40"
  "29571-hc0"
)

echo "Filtering directories in: $ROOT_DIR"

# Convert keep_dirs to associative array for fast lookup
declare -A keep_map
for dir in "${keep_dirs[@]}"; do
  keep_map["$dir"]=1
done

# Loop through all subdirectories and delete those not in keep_dirs
for dir in "$ROOT_DIR"/*/; do
  dir_name=$(basename "$dir")
  if [[ -z "${keep_map[$dir_name]}" ]]; then
    echo "Deleting: $dir_name"
    rm -rf "$dir"
  else
    echo "Keeping:  $dir_name"
  fi
done

echo "Cleanup complete."
