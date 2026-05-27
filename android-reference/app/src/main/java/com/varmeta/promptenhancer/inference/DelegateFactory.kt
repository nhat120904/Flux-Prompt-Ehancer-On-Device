package com.varmeta.promptenhancer.inference

import android.content.Context // kept for public API signature stability
import android.os.Build
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.MappedByteBuffer

private const val TAG = "DelegateFactory"

/**
 * Selects the best available TFLite hardware delegate at runtime.
 *
 * MUST be called from a background thread — GPU delegate initialization
 * is not permitted on the main thread.
 */
object DelegateFactory {

    /**
     * Returns the best [TfliteDelegate] according to [preference].
     * The returned delegate MUST be closed after the interpreter that
     * uses it is closed.
     *
     * @param modelBuffer Used to probe each candidate delegate. Will be rewound
     *                    before each probe.
     */
    fun select(
        @Suppress("UNUSED_PARAMETER") context: Context,
        preference: DelegatePreference,
        modelBuffer: MappedByteBuffer
    ): TfliteDelegate {
        return when (preference) {
            DelegatePreference.CPU -> {
                Log.i(TAG, "Delegate preference is CPU — using XNNPACK")
                TfliteDelegate.Cpu
            }
            DelegatePreference.NNAPI -> tryNnapi(modelBuffer) ?: TfliteDelegate.Cpu
            DelegatePreference.GPU  -> tryGpu(modelBuffer) ?: TfliteDelegate.Cpu
            DelegatePreference.AUTO ->
                tryNnapi(modelBuffer)
                    ?: tryGpu(modelBuffer)
                    ?: TfliteDelegate.Cpu
        }
    }

    // -------------------------------------------------------------------------
    // NNAPI
    // -------------------------------------------------------------------------

    private fun tryNnapi(modelBuffer: MappedByteBuffer): TfliteDelegate.Nnapi? {
        // NNAPI is available from API 27 (Android 8.1); min SDK here is 26.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O_MR1) {
            Log.i(TAG, "NNAPI unavailable: API level ${Build.VERSION.SDK_INT} < 27")
            return null
        }

        return try {
            val options = NnApiDelegate.Options().apply {
                // Disallow NNAPI's own CPU path — it is slower than XNNPACK.
                // If no real hardware accelerator is available, NNAPI will fail
                // and we fall back to GPU or CPU.
                setUseNnapiCpu(false)
                setAllowFp16(true)
            }
            val delegate = NnApiDelegate(options)
            if (probeDelegate(modelBuffer, delegate)) {
                Log.i(TAG, "NNAPI delegate accepted by model")
                TfliteDelegate.Nnapi(delegate)
            } else {
                delegate.close()
                null
            }
        } catch (t: Throwable) {
            Log.w(TAG, "NNAPI delegate creation failed: ${t.message}")
            null
        }
    }

    // -------------------------------------------------------------------------
    // GPU
    // -------------------------------------------------------------------------

    private fun tryGpu(modelBuffer: MappedByteBuffer): TfliteDelegate.Gpu? {
        CompatibilityList().use { compatList ->
            if (!compatList.isDelegateSupportedOnThisDevice) {
                Log.i(TAG, "GPU delegate not supported on this device")
                return null
            }
        }

        return try {
            // Use the no-arg constructor to avoid GpuDelegateFactory.Options
            // which is not shipped in tensorflow-lite-gpu 2.16.x AAR.
            val delegate = GpuDelegate()
            if (probeDelegate(modelBuffer, delegate)) {
                Log.i(TAG, "GPU delegate accepted by model")
                TfliteDelegate.Gpu(delegate)
            } else {
                delegate.close()
                null
            }
        } catch (t: Throwable) {
            Log.w(TAG, "GPU delegate creation failed: ${t.message}")
            null
        }
    }

    // -------------------------------------------------------------------------
    // Probe: create a throw-away interpreter to confirm the delegate works.
    // -------------------------------------------------------------------------

    private fun probeDelegate(
        modelBuffer: MappedByteBuffer,
        delegate: org.tensorflow.lite.Delegate
    ): Boolean {
        return try {
            modelBuffer.rewind()
            val opts = Interpreter.Options().apply { addDelegate(delegate) }
            Interpreter(modelBuffer, opts).use { /* probe only — discard immediately */ }
            true
        } catch (t: Throwable) {
            Log.w(TAG, "Delegate probe failed: ${t.message}")
            false
        }
    }
}
