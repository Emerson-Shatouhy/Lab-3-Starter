
package com.google.aiedge.examples.imageclassification

import android.content.Context
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.merge
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(
    private val imageClassificationHelper: ImageClassificationHelper,
    private val cloudInferenceHelper: CloudInferenceHelper
) : ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val imageClassificationHelper = ImageClassificationHelper(context)
                val cloudInferenceHelper = CloudInferenceHelper(context)
                return MainViewModel(imageClassificationHelper, cloudInferenceHelper) as T
            }
        }
    }

    private var classificationJob: Job? = null

    private val setting = MutableStateFlow(Setting())
        .apply {
            viewModelScope.launch {
                // this will be called when the setting state is updated
                collectLatest {
                    // Update on-device options
                    imageClassificationHelper.setOptions(
                        ImageClassificationHelper.Options(
                            model = it.model,
                            delegate = it.delegate,
                            resultCount = it.resultCount,
                            probabilityThreshold = it.threshold
                        )
                    )

                    // Update cloud options
                    cloudInferenceHelper.setOptions(
                        CloudInferenceHelper.Options(
                            serverUrl = it.serverUrl,
                            resultCount = it.resultCount,
                            probabilityThreshold = it.threshold
                        )
                    )

                    // Initialize the appropriate classifier
                    if (it.inferenceMode == InferenceMode.DEVICE) {
                        imageClassificationHelper.initClassifier()
                    }
                }
            }
        }

    // Combine error flows from both helpers
    private val errorMessage = MutableStateFlow<Throwable?>(null).also {
        viewModelScope.launch {
            merge(
                imageClassificationHelper.error,
                cloudInferenceHelper.error
            ).collect(it)
        }
    }

    // Combine classification results from both helpers
    val uiState: StateFlow<UiState> = combine(
        merge(
            imageClassificationHelper.classification,
            cloudInferenceHelper.classification
        ).stateIn(
            viewModelScope,
            SharingStarted.WhileSubscribed(5_000),
            ImageClassificationHelper.ClassificationResult(emptyList(), 0L)
        ),
        setting.filterNotNull(),
        errorMessage,
    ) { result, setting, error ->
        UiState(
            inferenceTime = result.inferenceTime,
            categories = result.categories,
            setting = setting,
            errorMessage = error?.message
        )
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState())

    /** Start classify an image.
     *  @param imageProxy contain `imageBitMap` and imageInfo as `image rotation degrees`.
     */
    fun classify(imageProxy: ImageProxy) {
        classificationJob = viewModelScope.launch {
            val bitmap = imageProxy.toBitmap()
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees

            when (setting.value.inferenceMode) {
                InferenceMode.DEVICE -> {
                    imageClassificationHelper.classify(bitmap, rotationDegrees)
                }
                InferenceMode.CLOUD -> {
                    cloudInferenceHelper.classify(bitmap, rotationDegrees)
                }
            }

            imageProxy.close()
        }
    }

    /** Stop current classification */
    fun stopClassify() {
        classificationJob?.cancel()
    }

    /** Set inference mode (on-device or cloud) */
    fun setInferenceMode(mode: InferenceMode) {
        setting.update { it.copy(inferenceMode = mode) }
    }

    /** Set server URL for cloud inference */
    fun setServerUrl(url: String) {
        setting.update { it.copy(serverUrl = url) }
    }

    /** Set [ImageClassificationHelper.Delegate] (CPU/NNAPI) for ImageClassificationHelper */
    fun setDelegate(delegate: ImageClassificationHelper.Delegate) {
        setting.update { it.copy(delegate = delegate) }
    }

    /** Set [ImageClassificationHelper.Model] for ImageClassificationHelper */
    fun setModel(model: ImageClassificationHelper.Model) {
        setting.update { it.copy(model = model) }
    }

    /** Set Number of output classes of the [ImageClassificationHelper.Model] */
    fun setNumberOfResult(numResult: Int) {
        setting.update { it.copy(resultCount = numResult) }
    }

    /** Set the threshold so the label can display score */
    fun setThreshold(threshold: Float) {
        setting.update { it.copy(threshold = threshold) }
    }

    /** Clear error message after it has been consumed */
    fun errorMessageShown() {
        errorMessage.update { null }
    }
}