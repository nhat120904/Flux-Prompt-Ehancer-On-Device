package com.varmeta.promptenhancer.inference

import org.tensorflow.lite.Delegate
import java.io.Closeable

/**
 * Wraps a live TFLite delegate object and owns its [close] lifecycle.
 * Interpreter must be closed BEFORE the delegate it references.
 */
sealed class TfliteDelegate(val name: String) : Closeable {

    class Nnapi(val delegate: Delegate) : TfliteDelegate("NNAPI") {
        override fun close() {
            try { (delegate as? Closeable)?.close() } catch (_: Throwable) {}
        }
    }

    class Gpu(val delegate: Delegate) : TfliteDelegate("GPU") {
        override fun close() {
            try { (delegate as? Closeable)?.close() } catch (_: Throwable) {}
        }
    }

    object Cpu : TfliteDelegate("CPU/XNNPACK") {
        override fun close() { /* nothing to release */ }
    }
}
