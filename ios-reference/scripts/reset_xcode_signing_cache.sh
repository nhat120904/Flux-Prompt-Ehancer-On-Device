#!/usr/bin/env bash
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_BASE="/tmp/xcode_signing_backup_${STAMP}"
mkdir -p "${BACKUP_BASE}/profiles" "${BACKUP_BASE}/deriveddata"

PROFILE_DIR="$HOME/Library/Developer/Xcode/UserData/Provisioning Profiles"
MOBILE_PROFILE_DIR="$HOME/Library/MobileDevice/Provisioning Profiles"
DERIVED_DIR="$HOME/Library/Developer/Xcode/DerivedData"

if [ -d "$PROFILE_DIR" ]; then
  shopt -s nullglob
  files=("$PROFILE_DIR"/*.mobileprovision)
  if [ ${#files[@]} -gt 0 ]; then
    mv "${files[@]}" "${BACKUP_BASE}/profiles/"
  fi
  shopt -u nullglob
fi

if [ -d "$MOBILE_PROFILE_DIR" ]; then
  shopt -s nullglob
  mfiles=("$MOBILE_PROFILE_DIR"/*.mobileprovision)
  if [ ${#mfiles[@]} -gt 0 ]; then
    mv "${mfiles[@]}" "${BACKUP_BASE}/profiles/"
  fi
  shopt -u nullglob
fi

if [ -d "$DERIVED_DIR" ]; then
  shopt -s nullglob
  dd=("$DERIVED_DIR"/PromptEnhancerApp-*)
  if [ ${#dd[@]} -gt 0 ]; then
    mv "${dd[@]}" "${BACKUP_BASE}/deriveddata/"
  fi
  shopt -u nullglob
fi

# Remove stale Xcode account/team preferences that can point to broken credentials.
defaults delete com.apple.dt.Xcode DVTDeveloperAccountManagerAppleIDLists >/dev/null 2>&1 || true
defaults delete com.apple.dt.Xcode IDEProvisioningTeamByIdentifier >/dev/null 2>&1 || true
defaults delete com.apple.dt.Xcode IDEProvisioningTeamManagerLastSelectedTeamID >/dev/null 2>&1 || true

echo "Backed up to: ${BACKUP_BASE}"
echo "Next:"
echo "1) Open Xcode > Settings > Accounts and re-add Apple ID if needed"
echo "2) Manage Certificates... create Apple Development cert"
echo "3) Open PromptEnhancerApp.xcodeproj and build again"
