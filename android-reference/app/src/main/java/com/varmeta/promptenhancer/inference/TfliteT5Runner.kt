package com.varmeta.promptenhancer.inference

import android.content.Context
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

class TfliteT5Runner(
    context: Context,
    private val modelAssetPath: String = MODEL_ASSET_PATH,
    private val delegatePreference: DelegatePreference = DelegatePreference.AUTO
) : Closeable {

    companion object {
        const val MODEL_ASSET_PATH = "model/decoder_init.tflite"
        const val DEFAULT_MAX_INPUT_TOKENS = 128
        const val DEFAULT_MAX_DECODER_TOKENS = 64
        private const val TAG = "TfliteT5Runner"
    }

    private val interpreter: Interpreter
    private val activeDelegate: TfliteDelegate

    /** Which hardware backend is actually executing inference. */
    val delegateName: String get() = activeDelegate.name

    private val inputIdsIndex: Int
    private val attentionMaskIndex: Int
    private val decoderInputIdsIndex: Int
    private val outputIndex: Int

    private var inputIdsTokenCapacity: Int = DEFAULT_MAX_INPUT_TOKENS
    private var attentionMaskTokenCapacity: Int = DEFAULT_MAX_INPUT_TOKENS
    private var maxInputTokens: Int = DEFAULT_MAX_INPUT_TOKENS
    private var maxDecoderTokens: Int = DEFAULT_MAX_DECODER_TOKENS

    init {
        val modelBuffer = loadModelFile(context, modelAssetPath)

        // DelegateFactory.select() must be called on a background thread.
        // This is guaranteed by PromptEnhancerEngine calling getOrCreateService()
        // inside withContext(Dispatchers.Default).
        val delegate = DelegateFactory.select(context, delegatePreference, modelBuffer)
        activeDelegate = delegate
        Log.i(TAG, "[$modelAssetPath] Using delegate: ${delegate.name}")

        val options = Interpreter.Options().apply {
            when (delegate) {
                is TfliteDelegate.Nnapi -> {
                    addDelegate(delegate.delegate)
                    setUseXNNPACK(false)
                    setNumThreads(1)
                }
                is TfliteDelegate.Gpu -> {
                    addDelegate(delegate.delegate)
                    setUseXNNPACK(false)
                    setNumThreads(1)
                }
                TfliteDelegate.Cpu -> {
                    setUseXNNPACK(true)
                    setNumThreads(max(2, Runtime.getRuntime().availableProcessors() / 2))
                }
            }
        }

        modelBuffer.rewind()
        interpreter = Interpreter(modelBuffer, options)

        inputIdsIndex = findInputIndex("input_ids")
        attentionMaskIndex = findInputIndex("attention_mask")
        decoderInputIdsIndex = findInputIndex("decoder_input_ids")
        outputIndex = 0

        tryResize2DInput(inputIdsIndex, DEFAULT_MAX_INPUT_TOKENS)
        tryResize2DInput(attentionMaskIndex, DEFAULT_MAX_INPUT_TOKENS)
        tryResize2DInput(decoderInputIdsIndex, DEFAULT_MAX_DECODER_TOKENS)
        interpreter.allocateTensors()
        ensureEncoderInputsAligned()

        inputIdsTokenCapacity = tokenCapacityForTensor(inputIdsIndex, DEFAULT_MAX_INPUT_TOKENS)
        attentionMaskTokenCapacity = tokenCapacityForTensor(attentionMaskIndex, DEFAULT_MAX_INPUT_TOKENS)
        maxInputTokens = min(inputIdsTokenCapacity, attentionMaskTokenCapacity)
        maxDecoderTokens = tokenCapacityForTensor(decoderInputIdsIndex, DEFAULT_MAX_DECODER_TOKENS)
    }

    fun maxInputTokens(): Int = maxInputTokens

    fun maxDecoderTokens(): Int = maxDecoderTokens

    fun decodeNextTokenLogits(
        inputIds: IntArray,
        attentionMask: IntArray,
        decoderInputIds: IntArray
    ): FloatArray {
        require(inputIds.isNotEmpty() && inputIds.size <= maxInputTokens) {
            "input_ids must be 1..$maxInputTokens"
        }
        require(attentionMask.isNotEmpty()) {
            "attention_mask must not be empty"
        }
        require(decoderInputIds.isNotEmpty() && decoderInputIds.size <= maxDecoderTokens) {
            "decoder_input_ids must be 1..$maxDecoderTokens"
        }

        val inputBuffers = Array<Any>(interpreter.inputTensorCount) { index ->
            makeZeroInputBufferForTensor(index)
        }
        inputBuffers[inputIdsIndex] = makeIntegerInputBuffer(inputIds, inputIdsIndex, inputIdsTokenCapacity)
        inputBuffers[attentionMaskIndex] = makeIntegerInputBuffer(
            attentionMask,
            attentionMaskIndex,
            attentionMaskTokenCapacity
        )
        inputBuffers[decoderInputIdsIndex] = makeIntegerInputBuffer(
            decoderInputIds,
            decoderInputIdsIndex,
            maxDecoderTokens
        )

        val outputTensor = interpreter.getOutputTensor(outputIndex)
        val outputBuffer = allocateOutputBuffer(outputTensor)
        val outputs = mutableMapOf<Int, Any>(outputIndex to outputBuffer)

        val t0 = System.nanoTime()
        try {
            interpreter.runForMultipleInputsOutputs(inputBuffers, outputs)
        } catch (t: Throwable) {
            throw IllegalStateException(
                buildString {
                    append("Interpreter invocation failed: ")
                    append(t.message ?: t.javaClass.simpleName)
                    append("\n")
                    append(buildIoDebugSummary())
                },
                t
            )
        }

        Log.v(TAG, "[${activeDelegate.name}] inference ${(System.nanoTime() - t0) / 1_000_000}ms seq=${decoderInputIds.size}")

        return extractLogits(
            buffer = outputBuffer,
            outputTensor = outputTensor,
            decoderTokenCount = decoderInputIds.size
        )
    }

    override fun close() {
        interpreter.close()    // interpreter must be closed before the delegate it references
        activeDelegate.close()
    }

    private fun findInputIndex(requiredName: String): Int {
        var normalizedMatch = -1
        for (i in 0 until interpreter.inputTensorCount) {
            val name = interpreter.getInputTensor(i).name()
            if (name == requiredName) {
                return i
            }

            // Some exports include prefixes/suffixes like "serving_default_input_ids:0".
            val normalized = name.substringAfterLast('/').substringBefore(':')
            if (normalized == requiredName) {
                if (normalizedMatch != -1) {
                    throw IllegalStateException(
                        "Ambiguous input tensor name for $requiredName: " +
                            "${interpreter.getInputTensor(normalizedMatch).name()} and $name"
                    )
                }
                normalizedMatch = i
            }
        }

        if (normalizedMatch != -1) {
            return normalizedMatch
        }
        throw IllegalStateException("Missing required input tensor: $requiredName")
    }

    private fun makeZeroInputBufferForTensor(index: Int): ByteBuffer {
        val tensor = interpreter.getInputTensor(index)
        val size = tensor.numBytes()
        return ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder()).apply {
            rewind()
        }
    }

    private fun makeIntegerInputBuffer(values: IntArray, tensorIndex: Int, fixedLength: Int): ByteBuffer {
        val tensor = interpreter.getInputTensor(tensorIndex)
        val dataType = tensor.dataType()
        val elementBytes = when (dataType) {
            DataType.INT32 -> Int.SIZE_BYTES
            DataType.INT64 -> Long.SIZE_BYTES
            else -> throw IllegalStateException(
                "Unsupported input dtype for tensor ${tensor.name()}: $dataType"
            )
        }

        val buffer = ByteBuffer.allocateDirect(fixedLength * elementBytes).order(ByteOrder.nativeOrder())
        for (i in 0 until fixedLength) {
            val value = if (i < values.size) values[i] else 0
            if (dataType == DataType.INT64) {
                buffer.putLong(value.toLong())
            } else {
                buffer.putInt(value)
            }
        }
        buffer.rewind()
        return buffer
    }

    private fun tokenCapacityForTensor(index: Int, fallback: Int): Int {
        val shape = interpreter.getInputTensor(index).shape()
        if (shape.size >= 2) {
            var product = 1L
            for (i in 1 until shape.size) {
                val dim = shape[i]
                if (dim <= 0) {
                    return fallback
                }
                product *= dim.toLong()
                if (product > Int.MAX_VALUE) {
                    return fallback
                }
            }
            if (product > 0) {
                return product.toInt()
            }
        }
        return fallback
    }

    private fun allocateOutputBuffer(outputTensor: org.tensorflow.lite.Tensor): ByteBuffer {
        val dataType = outputTensor.dataType()
        val elementBytes = when (dataType) {
            DataType.FLOAT32 -> 4
            // Some LiteRT builds expose FP16 output without DataType.FLOAT16 enum.
            else -> {
                val vocab = inferOutputVocabSize(outputTensor)
                val minBytesPerElement = if (vocab > 0) outputTensor.numBytes() / vocab else 0
                if (minBytesPerElement == 2) 2 else 4
            }
        }

        val vocabSize = inferOutputVocabSize(outputTensor)
        val stepsCapacity = maxDecoderTokens
        val bytes = stepsCapacity * vocabSize * elementBytes

        return ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())
    }

    private fun inferOutputVocabSize(outputTensor: org.tensorflow.lite.Tensor): Int {
        val shape = outputTensor.shape()
        if (shape.isEmpty()) {
            throw IllegalStateException("Invalid output shape: ${shape.contentToString()}")
        }
        val vocab = shape.last()
        if (vocab <= 0) {
            throw IllegalStateException("Invalid vocab size in output shape: ${shape.contentToString()}")
        }
        return vocab
    }

    private fun ensureEncoderInputsAligned() {
        var inputCap = tokenCapacityForTensor(inputIdsIndex, DEFAULT_MAX_INPUT_TOKENS)
        var maskCap = tokenCapacityForTensor(attentionMaskIndex, DEFAULT_MAX_INPUT_TOKENS)
        if (inputCap == maskCap) {
            return
        }

        // Prefer longer context first, then fallback to shorter if graph constraints reject it.
        val candidates = listOf(max(inputCap, maskCap), min(inputCap, maskCap)).distinct()
        for (candidate in candidates) {
            val resizedInput = tryResize2DInput(inputIdsIndex, candidate)
            val resizedMask = tryResize2DInput(attentionMaskIndex, candidate)
            if (resizedInput || resizedMask) {
                interpreter.allocateTensors()
            }

            inputCap = tokenCapacityForTensor(inputIdsIndex, DEFAULT_MAX_INPUT_TOKENS)
            maskCap = tokenCapacityForTensor(attentionMaskIndex, DEFAULT_MAX_INPUT_TOKENS)
            if (inputCap == maskCap) {
                return
            }
        }

        throw IllegalStateException(
            "Encoder input shapes are inconsistent and cannot be aligned: " +
                "input_ids=${interpreter.getInputTensor(inputIdsIndex).shape().contentToString()}, " +
                "attention_mask=${interpreter.getInputTensor(attentionMaskIndex).shape().contentToString()}"
        )
    }

    private fun tryResize2DInput(index: Int, seqLen: Int): Boolean {
        val shape = interpreter.getInputTensor(index).shape()
        if (shape.size != 2 || seqLen <= 0) {
            return false
        }
        val dims = intArrayOf(shape[0], seqLen)
        return try {
            // strict=false lets TFLite resize even when signature is concrete.
            interpreter.resizeInput(index, dims, false)
            true
        } catch (_: Throwable) {
            false
        }
    }

    private fun buildIoDebugSummary(): String {
        val inputSummary = buildString {
            append("Inputs:\n")
            for (i in 0 until interpreter.inputTensorCount) {
                val t = interpreter.getInputTensor(i)
                append("  [")
                append(i)
                append("] ")
                append(t.name())
                append(" shape=")
                append(t.shape().contentToString())
                append(" type=")
                append(t.dataType())
                append(" bytes=")
                append(t.numBytes())
                append('\n')
            }
        }

        val outputSummary = buildString {
            append("Outputs:\n")
            for (i in 0 until interpreter.outputTensorCount) {
                val t = interpreter.getOutputTensor(i)
                append("  [")
                append(i)
                append("] ")
                append(t.name())
                append(" shape=")
                append(t.shape().contentToString())
                append(" type=")
                append(t.dataType())
                append(" bytes=")
                append(t.numBytes())
                append('\n')
            }
        }

        return inputSummary + outputSummary
    }

    private fun extractLogits(
        buffer: ByteBuffer,
        outputTensor: org.tensorflow.lite.Tensor,
        decoderTokenCount: Int
    ): FloatArray {
        val shape = outputTensor.shape()
        val rank = shape.size
        val dataType = outputTensor.dataType()
        val elementBytesFromType = when (dataType) {
            DataType.FLOAT32 -> 4
            else -> 0
        }

        val (steps, vocabSize) = when (rank) {
            3 -> shape[1] to shape[2]
            2 -> 1 to shape[1]
            else -> throw IllegalStateException("Unexpected logits rank: ${shape.contentToString()}")
        }

        val bytesPerElement = when {
            elementBytesFromType > 0 -> elementBytesFromType
            steps > 0 && vocabSize > 0 -> buffer.capacity() / (steps * vocabSize)
            else -> 0
        }
        val stepsFromBuffer = if (vocabSize > 0 && bytesPerElement > 0) {
            buffer.capacity() / (vocabSize * bytesPerElement)
        } else {
            steps
        }
        val effectiveSteps = max(steps, stepsFromBuffer)

        val stepIndex = min(max(decoderTokenCount - 1, 0), max(effectiveSteps - 1, 0))
        val out = FloatArray(vocabSize)
        buffer.rewind()

        when {
            dataType == DataType.FLOAT32 || bytesPerElement == 4 -> {
                val floatBuffer = buffer.asFloatBuffer()
                val base = if (rank == 3) stepIndex * vocabSize else 0
                for (i in 0 until vocabSize) {
                    out[i] = floatBuffer.get(base + i)
                }
            }

            bytesPerElement == 2 -> {
                val shortBuffer = buffer.asShortBuffer()
                val base = if (rank == 3) stepIndex * vocabSize else 0
                for (i in 0 until vocabSize) {
                    out[i] = float16ToFloat(shortBuffer.get(base + i).toInt())
                }
            }

            else -> {
                throw IllegalStateException("Unsupported logits dtype: ${outputTensor.dataType()}")
            }
        }

        return out
    }

    private fun float16ToFloat(raw: Int): Float {
        val bits = raw and 0xFFFF
        val sign = bits ushr 15
        val exp = (bits ushr 10) and 0x1F
        val frac = bits and 0x03FF

        if (exp == 0) {
            if (frac == 0) {
                return if (sign == 0) 0f else -0f
            }
            var e = -1
            var f = frac
            while ((f and 0x0400) == 0) {
                f = f shl 1
                e -= 1
            }
            f = f and 0x03FF
            val outExp = 127 - 15 + 1 + e
            val outFrac = f shl 13
            val outBits = (sign shl 31) or (outExp shl 23) or outFrac
            return Float.fromBits(outBits)
        }

        if (exp == 31) {
            val outBits = (sign shl 31) or 0x7F800000 or (frac shl 13)
            return Float.fromBits(outBits)
        }

        val outExp = exp + (127 - 15)
        val outFrac = frac shl 13
        val outBits = (sign shl 31) or (outExp shl 23) or outFrac
        return Float.fromBits(outBits)
    }

    private fun loadModelFile(context: Context, assetPath: String): MappedByteBuffer {
        context.assets.openFd(assetPath).use { afd ->
            FileInputStream(afd.fileDescriptor).use { inputStream ->
                val channel = inputStream.channel
                return channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
            }
        }
    }
}
