package com.samsung.echogesture

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max

data class PredictionResult(
    val word: String,
    val confidence: Float,
    val top5: List<String>,
    val latencyMs: Float
)

class SignClassifier(private val context: Context) {

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private val vocabulary = mutableListOf<String>()

    init {
        loadVocabulary()
        initOrt()
    }

    private fun loadVocabulary() {
        try {
            context.assets.open("vocab.txt").use { stream ->
                BufferedReader(InputStreamReader(stream)).useLines { lines ->
                    lines.forEach { line ->
                        val word = line.trim().uppercase()
                        if (word.isNotEmpty()) {
                            vocabulary.add(word)
                        }
                    }
                }
            }
        } catch (_: Exception) {}

        if (vocabulary.isEmpty()) {
            vocabulary.addAll(listOf("COMPUTER", "HAPPY", "THANK YOU", "ORANGE", "HELP", "BEFORE", "CHANGE", "PLAY"))
        }
    }

    private fun initOrt() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val modelBytes = context.assets.open("model_int8.onnx").use { it.readBytes() }
            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
            }
            ortSession = ortEnv?.createSession(modelBytes, sessionOptions)
        } catch (_: Exception) {
            ortSession = null
        }
    }

    fun infer(featureArray: FloatArray): PredictionResult {
        val t0 = System.nanoTime()

        // Shape expected by Conv1D: (1, 320, 32) = 10,240 floats
        val totalElements = 1 * 320 * 32
        val byteBuffer = ByteBuffer.allocateDirect(totalElements * 4).order(ByteOrder.nativeOrder())
        val inputBuffer = byteBuffer.asFloatBuffer()

        if (featureArray.size >= totalElements) {
            inputBuffer.put(featureArray, 0, totalElements)
        } else {
            inputBuffer.put(featureArray)
            // Zero pad remainder
            for (i in featureArray.size until totalElements) {
                inputBuffer.put(0.0f)
            }
        }
        inputBuffer.flip()

        var topWord = "UNKNOWN"
        var confidence = 0.0f
        val top5 = mutableListOf<String>()

        val session = ortSession
        val env = ortEnv
        if (session != null && env != null) {
            var tensor: OnnxTensor? = null
            var output: OrtSession.Result? = null
            try {
                val inputName = session.inputNames.iterator().next()
                tensor = OnnxTensor.createTensor(env, inputBuffer, longArrayOf(1, 320, 32))
                output = session.run(mapOf(inputName to tensor))
                val rawLogits: FloatArray = when (val valObj = output[0].value) {
                    is Array<*> -> (valObj[0] as FloatArray)
                    is FloatArray -> valObj
                    else -> FloatArray(vocabulary.size) { 0f }
                }

                // Softmax
                var maxLogit = Float.NEGATIVE_INFINITY
                for (v in rawLogits) {
                    if (v > maxLogit) maxLogit = v
                }
                var sumExp = 0.0f
                val expLogits = FloatArray(rawLogits.size)
                for (i in rawLogits.indices) {
                    expLogits[i] = exp((rawLogits[i] - maxLogit).toDouble()).toFloat()
                    sumExp += expLogits[i]
                }
                val probs = FloatArray(rawLogits.size) { i -> expLogits[i] / sumExp }

                // Find top-5
                val sortedIndices = probs.indices.sortedByDescending { probs[it] }
                topWord = vocabulary.getOrElse(sortedIndices[0]) { "UNKNOWN" }
                confidence = probs[sortedIndices[0]]

                for (idx in sortedIndices.take(5)) {
                    top5.add(vocabulary.getOrElse(idx) { "UNKNOWN" })
                }
            } catch (_: Exception) {
                topWord = "UNKNOWN"
                confidence = 0.0f
                top5.clear()
                top5.add("UNKNOWN")
            } finally {
                tensor?.close()
                output?.close()
            }
        } else {
            topWord = "UNKNOWN"
            confidence = 0.0f
            top5.add("UNKNOWN")
        }

        // Measure true elapsed execution time without artificial floor
        val rawLatencyMs = ((System.nanoTime() - t0) / 1_000_000.0f)
        val roundedLatency = kotlin.math.round(rawLatencyMs * 100f) / 100f

        return PredictionResult(
            word = topWord,
            confidence = confidence,
            top5 = top5,
            latencyMs = roundedLatency
        )
    }

    fun close() {
        ortSession?.close()
        ortEnv?.close()
    }
}

