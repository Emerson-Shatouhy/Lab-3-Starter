/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.aiedge.examples.imageclassification

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.BottomSheetScaffold
import androidx.compose.material.Button
import androidx.compose.material.DropdownMenu
import androidx.compose.material.DropdownMenuItem
import androidx.compose.material.ExperimentalMaterialApi
import androidx.compose.material.Icon
import androidx.compose.material.MaterialTheme
import androidx.compose.material.RadioButton
import androidx.compose.material.RadioButtonDefaults
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.aiedge.examples.imageclassification.view.ApplicationTheme
import com.google.aiedge.examples.imageclassification.view.CameraScreen
import java.util.Locale


class MainActivity : ComponentActivity() {
    @OptIn(ExperimentalMaterialApi::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }
        setContent {

            val uiState by viewModel.uiState.collectAsStateWithLifecycle()

            LaunchedEffect(uiState.errorMessage) {
                if (uiState.errorMessage != null) {
                    Toast.makeText(
                        this@MainActivity, "${uiState.errorMessage}", Toast.LENGTH_SHORT
                    ).show()
                    viewModel.errorMessageShown()
                }
            }
            ApplicationTheme {
                BottomSheetScaffold(sheetPeekHeight = (90 + 20 * uiState.categories.size).dp,
                    sheetContent = {
                        BottomSheet(uiState = uiState, onModelSelected = {
                            viewModel.setModel(it)
                        }, onDelegateSelected = {
                            viewModel.setDelegate(it)
                        }, onThresholdSet = {
                            viewModel.setThreshold(it)
                        }, onMaxResultSet = {
                            viewModel.setNumberOfResult(it)
                        })
                    }) {
                    Column {
                        Header()
                        CameraScreen( // camera screen
                            uiState = uiState,
                            onImageAnalyzed = { imageProxy ->
                                viewModel.classify(imageProxy)
                            }
                        )
                    }
                }
            }
        }
    }

    @Composable
    fun Header() {
        TopAppBar(
            backgroundColor = MaterialTheme.colors.secondary,
            title = {
                Image(
                    modifier = Modifier.size(120.dp),
//                    alignment = Alignment.CenterStart,
                    painter = painterResource(id = R.drawable.logo),
                    contentDescription = null,
                )
            },
        )
    }

    @Composable
    fun BottomSheet(
        uiState: UiState,
        modifier: Modifier = Modifier,
        onModelSelected: (ImageClassificationHelper.Model) -> Unit,
        onDelegateSelected: (ImageClassificationHelper.Delegate) -> Unit,
        onThresholdSet: (value: Float) -> Unit,
        onMaxResultSet: (value: Int) -> Unit,
    ) {
        val categories = uiState.categories
        val inferenceTime = uiState.inferenceTime
        val threshold = uiState.setting.threshold
        val resultCount = uiState.setting.resultCount
        Column(
            modifier = modifier.padding(horizontal = 20.dp, vertical = 5.dp)
        ) {
            Spacer(modifier = Modifier.height(20.dp))
            LazyColumn {
                items(key = {
                    categories[it].label
                }, count = categories.size) {
                    val category = categories[it]
                    Row {
                        Text(
                            modifier = Modifier.weight(0.5f),
                            text = if (category.score <= 0f) "--" else category.label,
                            fontSize = 15.sp,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            text = if (category.score <= 0f) "--" else String.format(
                                Locale.US, "%.2f", category.score
                            ),
                            fontSize = 15.sp, fontWeight = FontWeight.SemiBold,
                        )
                    }
                }
            }
            Image(
                modifier = Modifier
                    .size(40.dp)
                    .align(Alignment.CenterHorizontally),
                painter = painterResource(id = R.drawable.ic_chevron_up),
                colorFilter = ColorFilter.tint(MaterialTheme.colors.secondary),
                contentDescription = ""
            )
            Row {
                Text(
                    modifier = Modifier.weight(0.5f),
                    text = stringResource(id = R.string.inference_title)
                )
                Text(text = stringResource(id = R.string.inference_value, inferenceTime))
            }
            Spacer(modifier = Modifier.height(20.dp))
            ModelSelection(onModelSelected = {
                onModelSelected(it)
            })
            Spacer(modifier = Modifier.height(20.dp))
            OptionMenu(label = stringResource(id = R.string.delegate),
                options = ImageClassificationHelper.Delegate.entries.map { it.name }) {
                onDelegateSelected(ImageClassificationHelper.Delegate.valueOf(it))
            }
            Spacer(modifier = Modifier.height(10.dp))
            AdjustItem(
                name = stringResource(id = R.string.threshold),
                value = uiState.setting.threshold,
                onMinusClicked = {
                    if (threshold > 0.3f) {
                        val newThreshold = (threshold - 0.1f).coerceAtLeast(0.3f)
                        onThresholdSet(newThreshold)
                    }
                },
                onPlusClicked = {
                    if (threshold < 0.8f) {
                        val newThreshold = threshold + 0.1f.coerceAtMost(0.8f)
                        onThresholdSet(newThreshold)
                    }
                },
            )

            AdjustItem(
                name = stringResource(id = R.string.maxResult), value = resultCount,
                onMinusClicked = {
                    if (resultCount >= 2) {
                        val count = resultCount - 1
                        onMaxResultSet(count)
                    }
                },
                onPlusClicked = {
                    if (resultCount < 5) {
                        val count = resultCount + 1
                        onMaxResultSet(count)
                    }
                },
            )
        }
    }


    // custom UI that renders a dropdown menu with a label
    // It allows the user to select an option from a predefined list of options
    // and provides a callback (onOptionSelected) to notify when an option is selected.
    @Composable
    fun OptionMenu(
        label: String,
        modifier: Modifier = Modifier,
        options: List<String>,
        onOptionSelected: (option: String) -> Unit,
    ) {
        var expanded by remember { mutableStateOf(false) }
        var option by remember { mutableStateOf(options.first()) }
        Row(
            modifier = modifier, verticalAlignment = Alignment.CenterVertically
        ) {
            Text(modifier = Modifier.weight(0.5f), text = label, fontSize = 15.sp)
            Box {
                Row(
                    modifier = Modifier.clickable {
                        expanded = true
                    }, verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(text = option, fontSize = 15.sp)
                    Spacer(modifier = Modifier.width(5.dp))
                    Icon(
                        imageVector = Icons.Default.ArrowDropDown,
                        contentDescription = "Localized description"
                    )
                }

                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    options.forEach {
                        DropdownMenuItem(
                            content = {
                                Text(it, fontSize = 15.sp)
                            },
                            onClick = {
                                option = it
                                onOptionSelected(option)
                                expanded = false
                            },
                        )
                    }
                }
            }
        }
    }

    @Composable
    fun ModelSelection(
        modifier: Modifier = Modifier,
        onModelSelected: (ImageClassificationHelper.Model) -> Unit,
    ) {
        val radioOptions = ImageClassificationHelper.Model.entries.map { it.name }.toList()
        var selectedOption by remember { mutableStateOf(radioOptions.first()) }

        Column(modifier = modifier) {
            radioOptions.forEach { option ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    RadioButton(
                        colors = RadioButtonDefaults.colors(selectedColor = MaterialTheme.colors.primary),
                        selected = (option == selectedOption),
                        onClick = {
                            if (selectedOption == option) return@RadioButton
                            onModelSelected(ImageClassificationHelper.Model.valueOf(option))
                            selectedOption = option
                        }, // Recommended for accessibility with screen readers
                    )
                    Text(
                        modifier = Modifier.padding(start = 16.dp), text = option, fontSize = 15.sp
                    )
                }
            }
        }
    }

    @Composable
    fun AdjustItem(
        name: String,
        value: Number,
        modifier: Modifier = Modifier,
        onMinusClicked: () -> Unit,
        onPlusClicked: () -> Unit,
    ) {
        Row(
            modifier = modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                modifier = Modifier.weight(0.5f),
                text = name,
                fontSize = 15.sp,
            )
            Row(
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Button(
                    onClick = {
                        onMinusClicked()
                    }) {
                    Text(text = "-", fontSize = 15.sp, color = Color.White)
                }
                Spacer(modifier = Modifier.width(10.dp))
                Text(
                    modifier = Modifier.width(30.dp),
                    textAlign = TextAlign.Center,
                    text = if (value is Float) String.format(
                        Locale.US, "%.1f", value
                    ) else value.toString(),
                    fontSize = 15.sp,
                )
                Spacer(modifier = Modifier.width(10.dp))
                Button(
                    onClick = {
                        onPlusClicked()
                    }) {
                    Text(text = "+", fontSize = 15.sp, color = Color.White)
                }
            }
        }
    }
}