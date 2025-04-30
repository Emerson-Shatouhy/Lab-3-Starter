package com.google.aiedge.examples.imageclassification

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Base64
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

class CloudInferenceHelper(private val context: Context) {
    class Options(
        var serverUrl: String = DEFAULT_SERVER_URL,
        var resultCount: Int = DEFAULT_RESULT_COUNT,
        var probabilityThreshold: Float = DEFAULT_THRESHOLD
    )

    companion object {
        private const val TAG = "CloudInference"


        const val DEFAULT_SERVER_URL = "http://10.0.2.2:5000/predict"
        const val DEFAULT_RESULT_COUNT = 3
        const val DEFAULT_THRESHOLD = 0.3f
    }

    private var options = Options()
    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    /** As the result of cloud classification, this value emits map of probabilities */
    val classification: SharedFlow<ImageClassificationHelper.ClassificationResult>
        get() = _classification
    private val _classification = MutableSharedFlow<ImageClassificationHelper.ClassificationResult>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    val error: SharedFlow<Throwable?>
        get() = _error
    private val _error = MutableSharedFlow<Throwable?>()

    fun setOptions(options: Options) {
        this.options = options
    }

    suspend fun classify(bitmap: Bitmap, rotationDegrees: Int) {
        try {
            withContext(Dispatchers.IO) {
                val startTime = SystemClock.uptimeMillis()


                val outputStream = ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
                val imageBytes = outputStream.toByteArray()
                val base64Image = Base64.encodeToString(imageBytes, Base64.DEFAULT)


                val jsonRequest = JSONObject().apply {
                    put("image", base64Image)
                    put("rotation", rotationDegrees)
                }


                val mediaType = "application/json; charset=utf-8".toMediaType()
                val requestBody = jsonRequest.toString().toRequestBody(mediaType)
                val request = Request.Builder()
                    .url(options.serverUrl)
                    .post(requestBody)
                    .build()


                client.newCall(request).execute().use { response ->
                    if (!response.isSuccessful) {
                        throw Exception("Server error: ${response.code}")
                    }


                    val jsonResponse = JSONObject(response.body?.string() ?: "")
                    val predictionsArray = jsonResponse.getJSONArray("predictions")

                    // Convert predictions to categories
                    val categories = mutableListOf<ImageClassificationHelper.Category>()
                    for (i in 0 until predictionsArray.length()) {
                        val prediction = predictionsArray.getJSONObject(i)
                        val label = prediction.getString("label")
                        val score = prediction.getDouble("probability").toFloat()

                        if (score >= options.probabilityThreshold) {
                            categories.add(ImageClassificationHelper.Category(label, score))
                        }
                    }


                    val sortedCategories = categories.sortedByDescending { it.score }
                        .take(options.resultCount)

                    val inferenceTime = SystemClock.uptimeMillis() - startTime
                    if (isActive) {
                        _classification.emit(
                            ImageClassificationHelper.ClassificationResult(
                                sortedCategories,
                                inferenceTime
                            )
                        )
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Cloud inference error occurred: ${e.message}")
            _error.emit(e)
        }
    }
}