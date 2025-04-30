package com.google.aiedge.examples.imageclassification

data class UiState(
    val inferenceTime: Long = 0L,
    val categories: List<ImageClassificationHelper.Category> = List(3) {
        ImageClassificationHelper.Category("", 0f)
    },
    val setting: Setting = Setting(),
    val errorMessage: String? = null
)

enum class InferenceMode {
    DEVICE, CLOUD
}

data class Setting(
    val model: ImageClassificationHelper.Model = ImageClassificationHelper.DEFAULT_MODEL,
    val delegate: ImageClassificationHelper.Delegate = ImageClassificationHelper.DEFAULT_DELEGATE,
    val threshold: Float = ImageClassificationHelper.DEFAULT_THRESHOLD,
    val resultCount: Int = ImageClassificationHelper.DEFAULT_RESULT_COUNT,
    val inferenceMode: InferenceMode = InferenceMode.DEVICE,
    val serverUrl: String = CloudInferenceHelper.DEFAULT_SERVER_URL
)