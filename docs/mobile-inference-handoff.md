# Mobile Inference Handoff (Android + iOS)

Tài liệu này dùng để handoff cho Android/iOS developer triển khai inference model `imranali291/flux-prompt-enhancer` on-device.

## 1. Artifacts cần có

### Android (LiteRT / TFLite)

Đặt trong app assets:

- `model/decoder_init.tflite`
- `model/decoder_with_past.tflite` (optional cho v1, chưa dùng)
- `model/encoder.tflite` (optional cho v1, chưa dùng)
- `model/conversion-metadata.json`
- `tokenizer/tokenizer_vocab.json`
- các file tokenizer liên quan (`spiece.model`, `tokenizer_config.json`, `special_tokens_map.json`, ...)

### iOS (Core ML)

Bundle vào app:

- `AssetsData/Model/decoder_init.mlpackage`
- `AssetsData/Model/decoder_with_past.mlpackage` (optional cho v1)
- `AssetsData/Model/encoder.mlpackage` (optional cho v1)
- `AssetsData/Tokenizer/tokenizer_vocab.json`
- các file tokenizer liên quan

## 2. Decode contract chung

Dùng cùng contract để giữ parity Android/iOS:

- `top_p = 0.92`
- `temperature = 0.8`
- `repetition_penalty = 1.2`
- `no_repeat_ngram_size = 3`
- `min_new_tokens = 60`
- `max_new_tokens = 63` (cap theo graph decoder length = 64, trừ token start)
- `retries = 3`
- `timeout_ms = 8000`

Finish reason:

- `eos`: text kết thúc bằng `.`, `!`, `?`
- `max_tokens`: hết budget token
- `timeout`: quá timeout

## 3. Android integration sample

### Gradle dependencies

```kotlin
implementation("org.tensorflow:tensorflow-lite:2.16.1")
implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.16.1")
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
```

### Gọi inference từ ViewModel/Activity

```kotlin
private val engine by lazy { PromptEnhancerEngine(applicationContext) }

lifecycleScope.launch {
    val result = engine.enhance(
        prompt = "beautiful house with text 'hello'",
        options = EnhanceOptions(
            topP = 0.92f,
            temperature = 0.8f,
            maxNewTokens = 63
        )
    )

    // result.text
    // result.latencyMs
    // result.tokensGenerated
    // result.finishReason
}
```

### Reference implementation trong repo

- Engine: `android-reference/app/src/main/java/com/varmeta/promptenhancer/inference/PromptEnhancerEngine.kt`
- Service decode loop: `android-reference/app/src/main/java/com/varmeta/promptenhancer/inference/PromptEnhancerService.kt`
- LiteRT runner: `android-reference/app/src/main/java/com/varmeta/promptenhancer/inference/TfliteT5Runner.kt`
- Tokenizer: `android-reference/app/src/main/java/com/varmeta/promptenhancer/tokenizer/SentencePieceTokenizer.kt`

## 4. iOS integration sample

### Gọi inference từ Swift concurrency

```swift
let engine = PromptEnhancerEngine()

Task {
    do {
        let result = try await engine.enhance(
            prompt: "beautiful house with text 'hello'",
            options: EnhanceOptions(
                topP: 0.92,
                temperature: 0.8,
                maxNewTokens: 63
            )
        )

        // result.text
        // result.latencyMs
        // result.tokensGenerated
        // result.finishReason
    } catch {
        print("Enhance failed: \(error)")
    }
}
```

### Reference implementation trong repo

- Engine: `ios-reference/PromptEnhancerApp/Sources/PromptEnhancerEngine.swift`
- Service decode loop: `ios-reference/PromptEnhancerApp/Sources/CoreMLPromptEnhancerService.swift`
- Core ML runner: `ios-reference/PromptEnhancerApp/Sources/LocalCoreMLT5Runner.swift`
- Tokenizer: `ios-reference/PromptEnhancerApp/Sources/SentencePieceTokenizer.swift`

## 5. I/O shape contract (v1)

- `input_ids`: `[1, source_len]`, int32
- `attention_mask`: `[1, source_len]`, int32
- `decoder_input_ids`: `[1, decoder_len]`, int32
- output logits: `[1, decoder_len, vocab]`, float32/float16

Ghi chú:

- `source_len` hiện dùng 128.
- `decoder_len` cap ở 64.
- Mỗi bước decode đọc logits tại `stepIndex = decoder_input_ids.count - 1`.

## 6. Checklist debug nhanh

1. Tensor name mapping phải đúng exact name (`input_ids`, `attention_mask`, `decoder_input_ids`).
2. `input_ids` và `attention_mask` phải cùng `source_len`.
3. Output buffer phải đủ cho full tensor `[1, decoder_len, vocab]`.
4. Tokenizer phải đúng file `tokenizer_vocab.json` cùng model.
5. Nếu lệch output giữa Android/iOS, kiểm tra lại thứ tự post-processing logits:
   - repetition penalty
   - no-repeat-ngram
   - temperature
   - top-p sampling

## 7. Handoff message mẫu cho mobile team

"Please implement on-device prompt enhancement using the shared decode contract in `docs/inference-spec.md` and integration notes in this file. Use the reference runners/services in `android-reference` and `ios-reference` to keep Android/iOS output behavior aligned."
