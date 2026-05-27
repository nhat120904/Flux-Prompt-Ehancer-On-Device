package com.varmeta.promptenhancer.inference

/**
 * Controls which hardware delegate TfliteT5Runner will attempt to use.
 *
 * AUTO  - try NNAPI, then GPU, then fall back to CPU (XNNPACK). Default.
 * NNAPI - try NNAPI only, then fall back to CPU.
 * GPU   - try GPU only, then fall back to CPU.
 * CPU   - always use XNNPACK multithreaded CPU (original behavior).
 */
enum class DelegatePreference {
    AUTO,
    NNAPI,
    GPU,
    CPU
}
