# iOS App Reference (SwiftUI + Core ML)

This folder now contains a runnable iOS app project:

- `PromptEnhancerApp.xcodeproj`
- `PromptEnhancerApp/` (SwiftUI app source + bundled model/tokenizer resources)

The app runs on-device prompt enhancement using your exported Core ML artifacts from `build/ios-export`.

## Open and run in Xcode

1. Open `ios-reference/PromptEnhancerApp.xcodeproj` in Xcode.
2. Select your Apple Team in **Signing & Capabilities** for target `PromptEnhancerApp`.
3. Connect iPhone and choose it as Run destination.
4. Build & Run.

## Troubleshooting signing/build

If you see `Command CodeSign failed with a nonzero exit code`:

1. In target `PromptEnhancerApp` -> **Signing & Capabilities**:
   - Enable **Automatically manage signing**
   - Select your Apple Team
2. Ensure the Bundle Identifier is unique in your team (for example `com.<yourname>.promptenhancer`).
3. In Xcode: `Product` -> `Clean Build Folder`, then build again.
4. If needed, restart Xcode after signing changes.

If the error contains:

- `code object is not signed at all`
- `In subcomponent: .../embedded.mobileprovision`

make sure bundled assets are under `PromptEnhancerApp/AssetsData` (not a top-level `Resources` folder in the app bundle). On Xcode 26 / iOS 26, a top-level `Resources` folder can cause invalid bundle / codesign failures around `embedded.mobileprovision`.
After pulling latest changes, clean and rebuild.

If build logs mention:

- `No Accounts: Add a new account in Accounts settings`
- `Invalid credentials in keychain ... missing Xcode-Username`
- `Failed to load profile ... missing UUID property`

then your local Xcode signing state is broken (account/profile cache), not app code.
Fix sequence:

1. Xcode -> Settings -> Accounts: remove broken Apple ID, then add it again.
2. In the same screen: **Manage Certificates...** -> create `Apple Development` certificate if missing.
3. Close Xcode.
4. Delete or move all files in:
   `~/Library/Developer/Xcode/UserData/Provisioning Profiles/`
5. Re-open Xcode, open project, select your Team again, then Build.

You can also reset caches via script:

```bash
cd ios-reference
./scripts/reset_xcode_signing_cache.sh
```

The script clears:

- `~/Library/Developer/Xcode/UserData/Provisioning Profiles`
- `~/Library/MobileDevice/Provisioning Profiles`
- stale Xcode account/team preference keys in `com.apple.dt.Xcode`

## Troubleshooting install to device

If build succeeds but install fails with:

- `Failed to install the app on the device (CoreDeviceError 3002)`
- `This device has reached the maximum number of installed apps using a free developer profile`

then this is Apple free-profile quota (not app code). Remove one previously installed dev app from the same team, then install again.

CLI example:

```bash
xcrun devicectl list devices
xcrun devicectl device uninstall app --device <DEVICE_ID> <BUNDLE_ID_TO_REMOVE>
```

## Refresh assets after re-export

When you export a new iOS model bundle, run:

```bash
cd ios-reference
./scripts/sync_assets.sh
```

Default sources used by the script:

- Core ML export: `../build/ios-export`
- Tokenizer snapshot cache:
  `$HOME/.cache/huggingface/hub/models--imranali291--flux-prompt-enhancer/snapshots/eaeb982805f1a755d301cb4dbe1f3773230e97c6`

You can pass custom paths:

```bash
./scripts/sync_assets.sh /path/to/ios-export /path/to/hf-snapshot
```

## Notes

- Runtime: Core ML only.
- Current decode path: non-KV cache iterative decoding (`decoder_init.mlpackage`).
- Max decoder context is fixed at 64 tokens by exported graph.
