package com.samsung.echogesture

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.acos
import kotlin.math.sqrt

data class RawFrameLandmarks(
    val pose: Array<FloatArray>?,      // 6 x 3 (Nose, L_Sh, R_Sh, L_Elb, R_Elb, L_Wr)
    val leftHand: Array<FloatArray>?,  // 21 x 3
    val rightHand: Array<FloatArray>?  // 21 x 3
)

class FeatureExtractor(private val context: Context) {

    private var handLandmarker: HandLandmarker? = null
    private var poseLandmarker: PoseLandmarker? = null
    // Frame buffer capacity for up to 4.5 seconds at 30 FPS (~135 frames)
    private val frameBuffer = ArrayDeque<RawFrameLandmarks>(140)

    init {
        initHandLandmarker()
        initPoseLandmarker()
    }

    private fun initHandLandmarker() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setMinHandDetectionConfidence(0.3f)
                .setMinHandPresenceConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .setNumHands(2)
                .setRunningMode(RunningMode.IMAGE)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(context, options)
        } catch (_: Exception) {
            handLandmarker = null
        }
    }

    private fun initPoseLandmarker() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("pose_landmarker_lite.task")
                .build()

            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setMinPoseDetectionConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .setNumPoses(1)
                .setRunningMode(RunningMode.IMAGE)
                .build()

            poseLandmarker = PoseLandmarker.createFromOptions(context, options)
        } catch (_: Exception) {
            poseLandmarker = null
        }
    }

    fun processFrame(bitmap: Bitmap): RawFrameLandmarks {
        val hLandmarker = handLandmarker
        val pLandmarker = poseLandmarker

        if (hLandmarker == null && pLandmarker == null) {
            val empty = RawFrameLandmarks(null, null, null)
            recordFrame(empty)
            return empty
        }

        return try {
            val mpImage = BitmapImageBuilder(bitmap).build()

            // 1. Real 3D Hand Landmarks (21 points per hand)
            var left: Array<FloatArray>? = null
            var right: Array<FloatArray>? = null

            if (hLandmarker != null) {
                val handResult: HandLandmarkerResult = hLandmarker.detect(mpImage)
                val landmarksList = handResult.landmarks()
                val handednessList = handResult.handednesses()

                for (i in landmarksList.indices) {
                    val lms = landmarksList[i]
                    val handPts = Array(21) { FloatArray(3) }
                    for (j in 0 until 21.coerceAtMost(lms.size)) {
                        handPts[j][0] = lms[j].x()
                        handPts[j][1] = lms[j].y()
                        handPts[j][2] = lms[j].z()
                    }

                    var isRight = true
                    if (i < handednessList.size && handednessList[i].isNotEmpty()) {
                        val label = handednessList[i][0].categoryName()
                        isRight = label.equals("Right", ignoreCase = true)
                    }

                    // Preserve both hands if MediaPipe assigns identical labels to both hands
                    if (isRight) {
                        if (right == null) {
                            right = handPts
                        } else if (left == null) {
                            left = handPts
                        } else {
                            right = handPts
                        }
                    } else {
                        if (left == null) {
                            left = handPts
                        } else if (right == null) {
                            right = handPts
                        } else {
                            left = handPts
                        }
                    }
                }
            }

            // 2. Real 3D Pose Landmarks (6 key upper-body anchor points)
            var posePts: Array<FloatArray>? = null
            if (pLandmarker != null) {
                val poseResult: PoseLandmarkerResult = pLandmarker.detect(mpImage)
                val poseLmsList = poseResult.landmarks()
                if (poseLmsList.isNotEmpty() && poseLmsList[0].size >= 16) {
                    val lms = poseLmsList[0]
                    // MediaPipe Pose indices: 0: Nose, 11: L_Shoulder, 12: R_Shoulder, 13: L_Elbow, 14: R_Elbow, 15: L_Wrist
                    val keyIndices = intArrayOf(0, 11, 12, 13, 14, 15)
                    val extracted = Array(6) { FloatArray(3) }
                    for (k in 0 until 6) {
                        val idx = keyIndices[k]
                        extracted[k][0] = lms[idx].x()
                        extracted[k][1] = lms[idx].y()
                        extracted[k][2] = lms[idx].z()
                    }
                    posePts = extracted
                }
            }

            val frameData = RawFrameLandmarks(posePts, left, right)
            recordFrame(frameData)
            frameData
        } catch (_: Exception) {
            val empty = RawFrameLandmarks(null, null, null)
            recordFrame(empty)
            empty
        }
    }

    private fun recordFrame(frame: RawFrameLandmarks) {
        synchronized(frameBuffer) {
            if (frameBuffer.size >= 140) {
                frameBuffer.removeFirst()
            }
            frameBuffer.addLast(frame)
        }
    }

    fun hasDetectedHands(): Boolean {
        synchronized(frameBuffer) {
            return frameBuffer.any { it.leftHand != null || it.rightHand != null }
        }
    }

    fun hasDetectedPose(): Boolean {
        synchronized(frameBuffer) {
            return frameBuffer.any { it.pose != null }
        }
    }

    /**
     * Measures real-time kinematic velocity of wrists between recent frames.
     * Used for real-time continuous gesture boundary & stroke detection.
     */
    fun getCurrentWristMotion(): Float {
        synchronized(frameBuffer) {
            if (frameBuffer.size < 2) return 0.0f
            val curr = frameBuffer.last()
            val prev = frameBuffer[frameBuffer.size - 2]
            var totalMotion = 0.0f
            var count = 0
            if (curr.rightHand != null && prev.rightHand != null) {
                val dx = curr.rightHand[0][0] - prev.rightHand[0][0]
                val dy = curr.rightHand[0][1] - prev.rightHand[0][1]
                totalMotion += sqrt(dx * dx + dy * dy)
                count++
            }
            if (curr.leftHand != null && prev.leftHand != null) {
                val dx = curr.leftHand[0][0] - prev.leftHand[0][0]
                val dy = curr.leftHand[0][1] - prev.leftHand[0][1]
                totalMotion += sqrt(dx * dx + dy * dy)
                count++
            }
            return if (count > 0) totalMotion / count else 0.0f
        }
    }

    /**
     * Extracts rolling 32-frame normalized tensor for live stream continuous inference.
     */
    fun getRolling32FrameTensor(): FloatArray? {
        val snapshot = synchronized(frameBuffer) {
            if (frameBuffer.size < 20) return null
            frameBuffer.takeLast(32)
        }
        return buildTensorFromFrames(snapshot)
    }

    /**
     * Constructs the exact 320-dimensional normalized feature matrix expected by model_int8.onnx.
     * Shape: (1, 320, 32) = 10,240 floats.
     */
    fun buildNormalizedFeatureTensor(): FloatArray {
        val snapshot = synchronized(frameBuffer) { frameBuffer.toList() }
        // Take the latest ~45 frames (1.5 seconds) for single sign prediction
        val recent = if (snapshot.size > 45) snapshot.takeLast(45) else snapshot
        return buildTensorFromFrames(recent)
    }

    private fun buildTensorFromFrames(available: List<RawFrameLandmarks>): FloatArray {
        val samples = mutableListOf<RawFrameLandmarks>()
        if (available.isEmpty()) {
            repeat(32) { samples.add(RawFrameLandmarks(null, null, null)) }
        } else {
            // Resample or pad available frames to exactly 32
            for (i in 0 until 32) {
                val idx = ((i.toFloat() / 31f) * (available.size - 1)).toInt()
                samples.add(available[idx.coerceIn(0, available.size - 1)])
            }
        }

        val T = 32
        // featureMatrix[t][feat_dim] -> (32, 320)
        val matrix = Array(T) { FloatArray(320) }

        for (t in 0 until T) {
            val frame = samples[t]
            val left = frame.leftHand
            val right = frame.rightHand
            val pose = frame.pose

            // 1. Pose features (18-dim: 6 points x 3)
            var midX = 0.5f
            var midY = 0.6f
            var shoulderScale = 0.35f

            if (pose != null) {
                // Real anatomical measurements from MediaPipe Pose Landmarker
                val lSh = pose[1]
                val rSh = pose[2]
                midX = (lSh[0] + rSh[0]) / 2f
                midY = (lSh[1] + rSh[1]) / 2f
                val midZ = (lSh[2] + rSh[2]) / 2f
                val dx = lSh[0] - rSh[0]
                val dy = lSh[1] - rSh[1]
                val dz = lSh[2] - rSh[2]
                val span = sqrt(dx * dx + dy * dy + dz * dz)
                if (span > 0.05f) {
                    shoulderScale = span.coerceIn(0.15f, 0.65f)
                }

                for (p in 0 until 6) {
                    matrix[t][p * 3 + 0] = (pose[p][0] - midX) / shoulderScale
                    matrix[t][p * 3 + 1] = (pose[p][1] - midY) / shoulderScale
                    matrix[t][p * 3 + 2] = (pose[p][2] - midZ) / shoulderScale
                }
            } else {
                // Dynamic torso midpoint and shoulder scale estimation from detected hands
                if (left != null && right != null) {
                    midX = (left[0][0] + right[0][0]) / 2f
                    midY = ((left[0][1] + right[0][1]) / 2f + 0.15f).coerceIn(0.35f, 0.75f)
                    val dx = left[0][0] - right[0][0]
                    val dy = left[0][1] - right[0][1]
                    val d = sqrt(dx * dx + dy * dy)
                    if (d > 0.08f) {
                        shoulderScale = (d * 1.25f).coerceIn(0.25f, 0.55f)
                    }
                } else if (right != null) {
                    midX = (right[0][0] - 0.12f).coerceIn(0.2f, 0.8f)
                    midY = (right[0][1] + 0.15f).coerceIn(0.35f, 0.75f)
                } else if (left != null) {
                    midX = (left[0][0] + 0.12f).coerceIn(0.2f, 0.8f)
                    midY = (left[0][1] + 0.15f).coerceIn(0.35f, 0.75f)
                }

                val posePts = floatArrayOf(
                    midX, (midY - shoulderScale * 0.9f).coerceAtLeast(0.1f), 0.0f,
                    midX - shoulderScale / 2f, midY, 0.0f,
                    midX + shoulderScale / 2f, midY, 0.0f,
                    midX - shoulderScale / 2f - 0.08f, midY + 0.18f, 0.0f,
                    midX + shoulderScale / 2f + 0.08f, midY + 0.18f, 0.0f,
                    if (left != null) left[0][0] else midX - 0.2f,
                    if (left != null) left[0][1] else midY + 0.3f,
                    if (left != null) left[0][2] else 0.0f
                )

                for (p in 0 until 18) {
                    val coordVal = posePts[p]
                    val origin = if (p % 3 == 0) midX else if (p % 3 == 1) midY else 0f
                    matrix[t][p] = (coordVal - origin) / shoulderScale
                }
            }

            // 2. Hand local features (126-dim: 2 hands x 21 points x 3)
            var offset = 18

            // Left Hand (slot 0)
            if (left != null) {
                val wristX = left[0][0]
                val wristY = left[0][1]
                val wristZ = left[0][2]
                for (pt in 0 until 21) {
                    matrix[t][offset + pt * 3 + 0] = (left[pt][0] - wristX) / shoulderScale
                    matrix[t][offset + pt * 3 + 1] = (left[pt][1] - wristY) / shoulderScale
                    matrix[t][offset + pt * 3 + 2] = (left[pt][2] - wristZ) / shoulderScale
                }
            }
            offset += 63

            // Right Hand (slot 1)
            if (right != null) {
                val wristX = right[0][0]
                val wristY = right[0][1]
                val wristZ = right[0][2]
                for (pt in 0 until 21) {
                    matrix[t][offset + pt * 3 + 0] = (right[pt][0] - wristX) / shoulderScale
                    matrix[t][offset + pt * 3 + 1] = (right[pt][1] - wristY) / shoulderScale
                    matrix[t][offset + pt * 3 + 2] = (right[pt][2] - wristZ) / shoulderScale
                }
            }
            offset += 63

            // 3. Velocities (144-dim)
            if (t > 0) {
                for (v in 0 until 144) {
                    matrix[t][144 + v] = matrix[t][v] - matrix[t - 1][v]
                }
            }

            // 4. Presence masks (2-dim)
            matrix[t][288] = if (left != null) 1.0f else 0.0f
            matrix[t][289] = if (right != null) 1.0f else 0.0f

            // 5. Semantic distances (30-dim)
            var geomOffset = 290

            // Tip-to-thumb distances (8-dim: tips 8, 12, 16, 20)
            val tips = intArrayOf(8, 12, 16, 20)
            for (h in 0..1) {
                val hand = if (h == 0) left else right
                for (tip in tips) {
                    var d = 0.0f
                    if (hand != null) {
                        val dx = hand[tip][0] - hand[4][0]
                        val dy = hand[tip][1] - hand[4][1]
                        val dz = hand[tip][2] - hand[4][2]
                        d = sqrt(dx * dx + dy * dy + dz * dz) / shoulderScale
                    }
                    matrix[t][geomOffset++] = d
                }
            }

            // Tip-to-wrist distances (10-dim: tips 4, 8, 12, 16, 20)
            val allTips = intArrayOf(4, 8, 12, 16, 20)
            for (h in 0..1) {
                val hand = if (h == 0) left else right
                for (tip in allTips) {
                    var d = 0.0f
                    if (hand != null) {
                        val dx = hand[tip][0] - hand[0][0]
                        val dy = hand[tip][1] - hand[0][1]
                        val dz = hand[tip][2] - hand[0][2]
                        d = sqrt(dx * dx + dy * dy + dz * dz) / shoulderScale
                    }
                    matrix[t][geomOffset++] = d
                }
            }

            // Inter-hand distance (2-dim)
            // Column 308: Wrist-to-wrist (0.0 in wrist-relative space)
            matrix[t][geomOffset + 0] = 0.0f
            if (left != null && right != null) {
                // Column 309: Relative index-to-wrist distance difference between hands
                val dix = (left[8][0] - left[0][0]) - (right[8][0] - right[0][0])
                val diy = (left[8][1] - left[0][1]) - (right[8][1] - right[0][1])
                val diz = (left[8][2] - left[0][2]) - (right[8][2] - right[0][2])
                matrix[t][geomOffset + 1] = sqrt(dix * dix + diy * diy + diz * diz) / shoulderScale
            } else {
                matrix[t][geomOffset + 1] = 0.0f
            }
            geomOffset += 2

            // Curl angles (10-dim: 5 fingers x 2 hands)
            // Must strictly match Python features.py:
            // finger_joints = [(2, 3, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
            // v1 = pip - mcp, v2 = tip - pip
            // cos_curl = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
            val joints = arrayOf(
                intArrayOf(2, 3, 4),   // Thumb: MCP, IP, Tip
                intArrayOf(5, 6, 8),   // Index: MCP, PIP, Tip
                intArrayOf(9, 10, 12), // Middle: MCP, PIP, Tip
                intArrayOf(13, 14, 16),// Ring: MCP, PIP, Tip
                intArrayOf(17, 18, 20) // Pinky: MCP, PIP, Tip
            )
            for (h in 0..1) {
                val hand = if (h == 0) left else right
                for (j in joints) {
                    var cosCurl = 0.0f
                    if (hand != null) {
                        val mcp = j[0]
                        val pip = j[1]
                        val tip = j[2]

                        val v1x = hand[pip][0] - hand[mcp][0]
                        val v1y = hand[pip][1] - hand[mcp][1]
                        val v1z = hand[pip][2] - hand[mcp][2]

                        val v2x = hand[tip][0] - hand[pip][0]
                        val v2y = hand[tip][1] - hand[pip][1]
                        val v2z = hand[tip][2] - hand[pip][2]

                        val dot = v1x * v2x + v1y * v2y + v1z * v2z
                        val norm1 = sqrt(v1x * v1x + v1y * v1y + v1z * v1z) + 1e-6f
                        val norm2 = sqrt(v2x * v2x + v2y * v2y + v2z * v2z) + 1e-6f
                        cosCurl = (dot / (norm1 * norm2)).coerceIn(-1.0f, 1.0f)
                    }
                    matrix[t][geomOffset++] = cosCurl
                }
            }
        }

        // Transpose from (Time=32, Feat=320) to Conv1D format (1, Feat=320, Time=32)
        val tensor = FloatArray(1 * 320 * 32)
        for (f in 0 until 320) {
            for (t in 0 until 32) {
                tensor[f * 32 + t] = matrix[t][f]
            }
        }
        return tensor
    }

    fun close() {
        handLandmarker?.close()
        poseLandmarker?.close()
    }
}

