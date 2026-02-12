# Android Reference App (TFLite)

Project này là app Android chạy on-device model prompt enhancer bằng artifacts tại:

- `/Users/nhatcuong/code_project/varmeta/prompt-enhancer-on-device/build/android-export`

Code đã được đặt đầy đủ trong:

- `/Users/nhatcuong/code_project/varmeta/prompt-enhancer-on-device/android-reference`

## Có gì trong app

- UI nhập prompt, chạy enhance, hiển thị output + latency/tokens.
- Inference TFLite với `decoder_init.tflite`.
- Sampling logic giống iOS reference:
  - top-p
  - temperature
  - repetition penalty
  - no-repeat ngram
  - timeout/retry
- SentencePiece tokenizer đọc từ `tokenizer_vocab.json`.

## Assets đã bundle sẵn

### Model assets

- `app/src/main/assets/model/decoder_init.tflite`
- `app/src/main/assets/model/decoder_with_past.tflite`
- `app/src/main/assets/model/encoder.tflite`
- `app/src/main/assets/model/conversion-metadata.json`

### Tokenizer assets

- `app/src/main/assets/tokenizer/tokenizer_vocab.json`
- các file tokenizer liên quan khác (`spiece.model`, `tokenizer_config.json`, ...)

## Mở bằng Android Studio và chạy

1. Mở folder:
   - `/Users/nhatcuong/code_project/varmeta/prompt-enhancer-on-device/android-reference`
2. Chờ Gradle sync.
3. Chọn emulator hoặc máy Android thật.
4. Run app.

## Build bằng terminal

```bash
cd /Users/nhatcuong/code_project/varmeta/prompt-enhancer-on-device/android-reference
./gradlew assembleDebug
```

APK debug sẽ nằm ở:

- `app/build/outputs/apk/debug/app-debug.apk`

## Cập nhật assets sau khi re-export model

```bash
cd /Users/nhatcuong/code_project/varmeta/prompt-enhancer-on-device/android-reference
./scripts/sync_assets.sh
```

Hoặc truyền path custom:

```bash
./scripts/sync_assets.sh /path/to/android-export /path/to/tokenizer-folder
```

## Ghi chú kỹ thuật

- Runtime: `org.tensorflow:tensorflow-lite` + `tensorflow-lite-select-tf-ops`.
- `aapt noCompress` đã bật cho `.tflite` để memory-map model từ assets.
- Decode path hiện dùng `decoder_init.tflite` (non-KV iterative decode).
