package com.samsung.echogesture

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.samsung.echogesture.databinding.ActivityMainBinding
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var classifier: SignClassifier
    private lateinit var featureExtractor: FeatureExtractor
    private lateinit var telemetryMonitor: TelemetryMonitor
    private var textToSpeech: TextToSpeech? = null

    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private val telemetryHandler = Handler(Looper.getMainLooper())

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission required for gesture recognition", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        classifier = SignClassifier(this)
        featureExtractor = FeatureExtractor(this)
        telemetryMonitor = TelemetryMonitor(this)
        cameraExecutor = Executors.newSingleThreadExecutor()
        textToSpeech = TextToSpeech(this, this)

        setupListeners()
        startTelemetryLoop()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun setupListeners() {
        binding.btnFlipCamera.setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                CameraSelector.LENS_FACING_BACK
            } else {
                CameraSelector.LENS_FACING_FRONT
            }
            startCamera()
        }

        binding.btnRecognizeSign.setOnClickListener {
            triggerSignRecognition()
        }

        binding.btnCompanionReply.setOnClickListener {
            showCompanionReplyDialog()
        }
    }

    private fun showCompanionReplyDialog() {
        val options = arrayOf(
            "Hello, nice to meet you!",
            "How can I help you today?",
            "I can understand your sign language.",
            "Thank you, see you tomorrow!",
            "Custom voice phrase..."
        )

        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("🎙️ Hearing Companion Voice Reply")
            .setItems(options) { _, which ->
                if (which == options.size - 1) {
                    val input = android.widget.EditText(this)
                    input.hint = "Enter companion speech..."
                    androidx.appcompat.app.AlertDialog.Builder(this)
                        .setTitle("Companion Voice")
                        .setView(input)
                        .setPositiveButton("Display on HUD") { _, _ ->
                            val txt = input.text.toString().trim()
                            if (txt.isNotEmpty()) displayCompanionSubtitle(txt)
                        }
                        .setNegativeButton("Cancel", null)
                        .show()
                } else {
                    displayCompanionSubtitle(options[which])
                }
            }
            .setNegativeButton("Dismiss", null)
            .show()
    }

    private fun displayCompanionSubtitle(text: String) {
        binding.tvCompanionSubtitle.text = "COMPANION [AUDIO TRANSCRIPTION]:\n\"$text\""
        binding.tvCompanionSubtitle.visibility = android.view.View.VISIBLE
        Handler(Looper.getMainLooper()).postDelayed({
            binding.tvCompanionSubtitle.visibility = android.view.View.GONE
        }, 8000L)
    }

    private fun triggerSignRecognition() {
        val originalText = binding.btnRecognizeSign.text
        binding.btnRecognizeSign.text = "RECORDING (1.5s)..."
        binding.btnRecognizeSign.setBackgroundColor(Color.parseColor("#FF9900"))

        binding.tvPrediction.text = "ANALYZING..."
        binding.tvConfidence.text = "Tracking 3D hand landmarks & INT8 forward pass..."

        Handler(Looper.getMainLooper()).postDelayed({
            val hasHands = featureExtractor.hasDetectedHands()

            if (!hasHands) {
                binding.tvPrediction.text = "NO HANDS IN VIEW"
                binding.tvConfidence.text = "Position hands in camera view • Model ready"
                binding.tvLatency.text = "0.00 ms\nLATENCY"
                speakAloud("Please position hands in camera view")
            } else {
                // Edge INT8 TCN classification
                val features = featureExtractor.buildNormalizedFeatureTensor()
                val result = classifier.infer(features)
                telemetryMonitor.recordInferenceEvent()
                val displayWord = if (result.confidence >= 0.15f && result.word != "UNKNOWN") {
                    result.word
                } else {
                    "UNCERTAIN GESTURE"
                }

                binding.tvPrediction.text = displayWord
                val confPct = (result.confidence * 100).toInt()
                binding.tvConfidence.text = "Confidence: $confPct% • Latency: ${result.latencyMs} ms • ✋ Hands Tracked"
                binding.tvLatency.text = "${result.latencyMs} ms\nLATENCY"
                speakAloud(displayWord)
            }

            binding.btnRecognizeSign.text = originalText
            binding.btnRecognizeSign.setBackgroundColor(Color.parseColor("#00E5FF"))
        }, 1500L)
    }

    private fun speakAloud(text: String) {
        textToSpeech?.speak(text.lowercase(Locale.ROOT), TextToSpeech.QUEUE_FLUSH, null, "EchoGestureTTS")
    }

    private fun startTelemetryLoop() {
        telemetryHandler.post(object : Runnable {
            override fun run() {
                val stats = telemetryMonitor.readCurrentTelemetry()
                binding.tvTemp.text = "${stats.tempDisplay}\nTEMP"
                binding.tvTemp.setTextColor(if (stats.isTempCompliant) Color.parseColor("#00FF88") else Color.parseColor("#FF3366"))

                binding.tvPower.text = "${stats.powerDisplay}\nSILICON"
                binding.tvPower.setTextColor(if (stats.isPowerCompliant) Color.parseColor("#00FF88") else Color.parseColor("#FF3366"))

                binding.tvRam.text = "${stats.ramDisplay}\nRAM"
                telemetryHandler.postDelayed(this, 1500)
            }
        })
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                var rawBitmap: Bitmap? = null
                var processedBitmap: Bitmap? = null
                try {
                    rawBitmap = imageProxy.toBitmap()
                    val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                    val isFront = (lensFacing == CameraSelector.LENS_FACING_FRONT)

                    // Correct camera sensor orientation & un-mirror selfie view
                    processedBitmap = if (rotationDegrees != 0 || isFront) {
                        val matrix = Matrix()
                        if (rotationDegrees != 0) {
                            matrix.postRotate(rotationDegrees.toFloat())
                        }
                        if (isFront) {
                            matrix.postScale(-1f, 1f, rawBitmap.width / 2f, rawBitmap.height / 2f)
                        }
                        Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
                    } else {
                        rawBitmap
                    }

                    featureExtractor.processFrame(processedBitmap)
                } catch (_: Exception) {
                } finally {
                    if (processedBitmap != null && processedBitmap !== rawBitmap && !processedBitmap.isRecycled) {
                        processedBitmap.recycle()
                    }
                    if (rawBitmap != null && !rawBitmap.isRecycled) {
                        rawBitmap.recycle()
                    }
                    imageProxy.close()
                }
            }

            val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
            } catch (exc: Exception) {
                Toast.makeText(this, "Failed to bind camera: ${exc.message}", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            textToSpeech?.language = Locale.US
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        telemetryHandler.removeCallbacksAndMessages(null)
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        classifier.close()
        featureExtractor.close()
    }
}
