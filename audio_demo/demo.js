const modelURLInput = document.getElementById('model-url');
const loadModelButton = document.getElementById('load-model');
const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
const mainCanvas = document.getElementById('main-canvas');
const spectrogramCanvas = document.getElementById('spectrogram-canvas');
const recogLabel = document.getElementById('recog-label');
const predictionCanvas = document.getElementById('prediction-canvas');
const transferPredictionCanvas = document.getElementById('transfer-prediction-canvas');
const transferLearnHistoryDiv = document.getElementById('transfer-learn-history');

let stopRequested = false;

let words;

// Setup slider for magnitude threshold.
const runOptions = {
  magnitudeThreshold: -35,
  magnitudeThresholdMin: -60,
  magnitudeThresholdMax: 0,
  sampleRate: 44100,
  frameSize: 1024,
  rotatingBufferSizeMultiplier: 2,
  refractoryPeriodMillis: 1000,
  waitingPeriodMillis: 500,
  numFrames: null,
  modelFFTLength: null,
  frameMillis: null,  // Frame duration in milliseconds.
};

setUpThresholdSlider(runOptions);

let intervalTask = null;

let model;

// Variables for transfer learning.
let transferWords;
let transferTensors = {};
let collectWordDivs = {};
let collectWordButtons = {};

loadModelButton.addEventListener('click', async () => {
  loadModelButton.disabled = true;
  await loadModelAndMetadataAndWarmUpModel(true);
  saveModelToLocalButton.disabled = false;
});

async function loadModelAndMetadataAndWarmUpModel(loadFromRemote) {
  const modelJSONSuffix = 'model.json';
  const metadataJSONSuffix = 'metadata.json';

  // 1. Load model.
  let loadModelFrom;
  if (loadFromRemote) {
    loadModelFrom = modelURLInput.value;
    if (loadModelFrom.indexOf(modelJSONSuffix) !==
        loadModelFrom.length - modelJSONSuffix.length) {
      alert(`Model URL must end in ${modelJSONSuffix}.`);
    }

    logToStatusDisplay('Loading model...');
  } else {
    loadModelFrom = LOCAL_MODEL_SAVE_LOCATION;
  }

  model = await tf.loadModel(loadModelFrom);
  model.summary();
  const inputShape = model.inputs[0].shape;
  runOptions.numFrames = inputShape[1];
  runOptions.modelFFTLength = inputShape[2];
  logToStatusDisplay(`numFrames: ${runOptions.numFrames}`);

  runOptions.frameMillis = runOptions.frameSize / runOptions.sampleRate * 1e3;

  console.assert(inputShape[3] === 1);

  // 2. Warm up the model.
  warmUpModel(3);

  // 3. Load the words and frameSize.
  let metadataJSON;
  if (loadFromRemote) {
    const loadMetadataFrom = loadModelFrom.slice(
        0, loadModelFrom.length - modelJSONSuffix.length) +
        metadataJSONSuffix;
    metadataJSON = await (await fetch(loadMetadataFrom)).json();
  } else {
    metadataJSON = JSON.parse(localStorage.getItem(MODEL_METADATA_SAVE_LOCATION));
  }

  if (runOptions.frameSize !== Number.parseInt(metadataJSON.frameSize)) {
    throw new Error(
      `Unexpected frame size from model: ${metadataJSON.frameSize}`);
  }
  words = metadataJSON.words;

  logToStatusDisplay('frameSize: ' + runOptions.frameSize);
  logToStatusDisplay(`Loaded ${words.length} words: ` + words);

  startButton.disabled = false;
  enterLearnWordsButton.disabled = false;
  startTransferLearnButton.disabled = false;

  // 4. If model has more than one heads, load the transfer words.
  if (model.outputs.length > 1) {
    transferWords =
        JSON.parse(localStorage.getItem(TRANSFER_WORDS_SAVE_LOCATION));
    learnWordsInput.value = transferWords.join(',');
    logToStatusDisplay(
        `Loaded transfer learned words: ${JSON.stringify(transferWords)}`);
  }
}

function warmUpModel(numPredictCalls) {
  const inputShape = model.inputs[0].shape;
  const x = tf.zeros([1].concat(inputShape.slice(1)));
  for (let i = 0; i < numPredictCalls; ++i) {
    const tBegin = new Date();
    model.predict(x);
    const tEnd = new Date();
    logToStatusDisplay(`Warm up ${i + 1} took: ${tEnd.getTime() - tBegin.getTime()} ms`);
  }
  x.dispose();
}

function start(collectOneSpeechSample) {
  stopRequested = false;
  navigator.mediaDevices.getUserMedia({audio: true, video: false})
    .then(stream => {
      logToStatusDisplay('getUserMedia() succeeded.');
      handleMicStream(stream, collectOneSpeechSample);
    }).catch(err => {
      logToStatusDisplay('getUserMedia() failed: ' + err.message);
    });
}

/**
 * Handle stream from obtained user media.
 * @param {*} stream
 * @param {*} collectOneSpeechSample
 */
function handleMicStream(stream, collectOneSpeechSample) {
  if (runOptions.numFrames == null || runOptions.modelFFTLength == null) {
    throw new Error('Load model first!');
  }

  const AudioContext = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioContext();
  logToStatusDisplay(`audioContext.sampleRate = ${audioContext.sampleRate}`);
  if (audioContext.sampleRate !== runOptions.sampleRate) {
    alert(
      `Mismatch in sampling rate: ` +
      `${audioContext.sampleRate} !== ${runOptions.sampleRate}`);
  }

  const source = audioContext.createMediaStreamSource(stream);

  const analyser = audioContext.createAnalyser();
  analyser.fftSize = runOptions.frameSize * 2;
  analyser.smoothingTimeConstant = 0.0;
  source.connect(analyser);

  const freqData = new Float32Array(analyser.frequencyBinCount);
  const rotatingBufferNumFrames =
    runOptions.numFrames * runOptions.rotatingBufferSizeMultiplier;
  const rotatingBufferSize =
    runOptions.modelFFTLength * rotatingBufferNumFrames;
  const rotatingBuffer = new Float32Array(rotatingBufferSize);

  let frameCount = 0;

  const frameDurationMillis =
    runOptions.frameSize / runOptions.sampleRate * 1e3;
  const waitingPeriodFrames = Math.round(
    runOptions.waitingPeriodMillis / frameDurationMillis);
  const refractoryPeriodFrames = Math.round(
    runOptions.refractoryPeriodMillis / frameDurationMillis);
  logToStatusDisplay(`waitingPeriodFrames: ${waitingPeriodFrames}`);
  logToStatusDisplay(`refractoryPeriodFrames: ${refractoryPeriodFrames}`);

  const tracker = new Tracker(waitingPeriodFrames, refractoryPeriodFrames);

  function onEveryAudioFrame() {
    if (stopRequested) {
      return;
    }

    analyser.getFloatFrequencyData(freqData);
    if (freqData[0] === -Infinity && freqData[1] === -Infinity) {
      // No signal from microphone. Do nothing.
      logToStatusDisplay('Warning: -Infinity magnitude.');
      return;
    }

    const freqDataSlice = freqData.slice(0, runOptions.modelFFTLength);
    plotSpectrum(mainCanvas, freqDataSlice, runOptions);

    const bufferPos = frameCount % rotatingBufferNumFrames;
    rotatingBuffer.set(freqDataSlice, bufferPos * runOptions.modelFFTLength);
    const spectralMax = getArrayMax(freqDataSlice);

    tracker.tick(spectralMax > runOptions.magnitudeThreshold);
    if (tracker.shouldFire()) {
      const freqData = getFrequencyDataFromRotatingBuffer(
        rotatingBuffer, frameCount - runOptions.numFrames);
      plotSpectrogram(
        spectrogramCanvas, freqData,
        runOptions.modelFFTLength, runOptions.modelFFTLength);
      const inputTensor = getInputTensorFromFrequencyData(freqData);

      if (collectOneSpeechSample) {
        stopRequested = true;
        clearInterval(intervalTask);

        if (transferTensors[collectOneSpeechSample] == null) {
          transferTensors[collectOneSpeechSample] = [];
        }
        transferTensors[collectOneSpeechSample].push(inputTensor);
        collectWordButtons[collectOneSpeechSample].textContent =
          `Collect "${collectOneSpeechSample}" sample ` +
          `(${transferTensors[collectOneSpeechSample].length})`;
        enableAllCollectWordButtons();
        const wordDiv = collectWordDivs[collectOneSpeechSample];
        const newCanvas = document.createElement('canvas');
        newCanvas.style['display'] = 'inline-block';
        newCanvas.style['vertical-align'] = 'middle';
        newCanvas.style['height'] = '100px';
        newCanvas.style['width'] = '150px';
        newCanvas.style['padding'] = '5px';
        wordDiv.appendChild(newCanvas);
        plotSpectrogram(
          newCanvas, freqData,
          runOptions.modelFFTLength, runOptions.modelFFTLength);
      } else {
        tf.tidy(() => {
          if (model.outputs.length === 1) {
            // No transfer learning has occurred; no transfer learned model
            // has been saved in IndexedDB.
            const probs = model.predict(inputTensor);
            plotPredictions(predictionCanvas, words, probs.dataSync());
            const recogIndex = tf.argMax(probs, -1).dataSync()[0];
            recogLabel.textContent += words[recogIndex] + ',';
          } else {
            // This is a two headed model from transfer learning.
            const probs = model.predict(inputTensor);
            const oldWordProbs = probs[0];
            const transferWordProbs = probs[1];
            plotPredictions(predictionCanvas, words, oldWordProbs.dataSync());
            const recogIndex = tf.argMax(oldWordProbs, -1).dataSync()[0];
            recogLabel.textContent += words[recogIndex] + ',';
            plotPredictions(
                transferPredictionCanvas, transferWords,
                transferWordProbs.dataSync());
          }
        });
        inputTensor.dispose();
      }
    } else if (tracker.isResting()) {
      // Clear prediction plot.
      plotPredictions(predictionCanvas);
      plotPredictions(transferPredictionCanvas);
    }

    frameCount++;
  }

  intervalTask = setInterval(
    onEveryAudioFrame,
    analyser.frequencyBinCount / audioContext.sampleRate * 1000);
}

function getArrayMax(xs) {
  let max = -Infinity;
  for (let i = 0; i < xs.length; ++i) {
    if (xs[i] > max) {
      max = xs[i];
    }
  }
  return max;
}

function getFrequencyDataFromRotatingBuffer(rotatingBuffer, frameCount) {
  const size = runOptions.numFrames * runOptions.modelFFTLength;
  const freqData = new Float32Array(size);

  const rotatingBufferSize = rotatingBuffer.length;
  const rotatingBufferNumFrames =
    rotatingBufferSize / runOptions.modelFFTLength;
  while (frameCount < 0) {
    frameCount += rotatingBufferNumFrames;
  }
  const indexBegin =
    (frameCount % rotatingBufferNumFrames) * runOptions.modelFFTLength;
  const indexEnd = indexBegin + size;

  for (let i = indexBegin; i < indexEnd; ++i) {
    freqData[ i - indexBegin]  = rotatingBuffer[i % rotatingBufferSize];
  }
  return freqData;
}

function getInputTensorFromFrequencyData(freqData) {
  const size = freqData.length;
  const tensorBuffer = tf.buffer([size]);
  for (let i = 0; i < freqData.length; ++i) {
    tensorBuffer.set(freqData[i], i);
  }
  return normalize(tensorBuffer.toTensor().reshape([
    1, runOptions.numFrames, runOptions.modelFFTLength, 1]));
}

startButton.addEventListener('click', () => {
  start();
  startButton.disabled = true;
  stopButton.disabled = false;
});

stopButton.addEventListener('click', () => {
  stopRequested = true;
  startButton.disabled = false;
  stopButton.disabled = true;
});

// UI code foro transfer learning.
const learnWordsInput = document.getElementById('learn-words');
const enterLearnWordsButton = document.getElementById('enter-learn-words');
const collectButtonsDiv = document.getElementById('collect-words');
const startTransferLearnButton =
  document.getElementById('start-transfer-learn');

enterLearnWordsButton.addEventListener('click', () => {
  enterLearnWordsButton.disabled = true;
  transferWords =
    learnWordsInput.value.trim().split(',').map(w => w.trim());

  for (const word of transferWords) {
    const wordDiv = document.createElement('div');
    wordDiv.style['border'] = 'solid 1px'
    const button = document.createElement('button');
    button.style['display'] = 'inline-block';
    button.style['vertical-align'] = 'middle';
    button.textContent = `Collect "${word}" sample (0)`;
    wordDiv.appendChild(button);
    wordDiv.style['height'] = '100px';
    collectButtonsDiv.appendChild(wordDiv);
    collectWordDivs[word] = wordDiv;
    collectWordButtons[word] = button;

    button.addEventListener('click', () => {
      disableAllCollectWordButtons();
      currentlyCollectedWord = word;
      logToStatusDisplay(
          `Collect one sample of word "${currentlyCollectedWord}"`);
      start(word);
    });
  }
});

startTransferLearnButton.addEventListener('click', async () => {
  const [xs, ys] = prepareLearnTensors();
  await doTransferLearning(xs, ys);
});

function disableAllCollectWordButtons() {
  for (const word in collectWordButtons) {
    collectWordButtons[word].disabled = true;
  }
}

function enableAllCollectWordButtons() {
  for (const word in collectWordButtons) {
    collectWordButtons[word].disabled = false;
  }
}

/**
 * @returns
 *   1. xs: A Tensor of shape `[numExamples, numTimeSteps, numFreqSteps]`.
 *   2. ys: A one-hot encoded target Tensor of shape `[numExamples, numWords]`.
 */
function prepareLearnTensors() {
  return tf.tidy(() => {
    const numDistinctWords = transferWords.length;
    let numWords = 0;
    let xs;
    let ys;
    for (let i = 0; i < transferWords.length; ++i) {
      const word = transferWords[i];
      for (const tensor of transferTensors[word]) {
        const yBuffer = tf.buffer([1, numDistinctWords]);
        yBuffer.set(1, 0, i);

        if (numWords === 0) {
          xs = tensor;
          ys = yBuffer.toTensor();
        } else {
          xs = tf.concat([xs, tensor], 0);
          ys = tf.concat([ys, yBuffer.toTensor()], 0);
        }

        numWords++;
      }
    }

    return [xs, ys];
  });
}

async function doTransferLearning(xs, ys) {
  const cutoffLayerIndex = 9;
  for (let i = 0; i <= cutoffLayerIndex; ++i) {
    model.layers[i].trainable = false;
  }

  const cutoffTensor = model.layers[cutoffLayerIndex].output;
  const newDenseLayer = tf.layers.dense({
    units: transferWords.length,
    activation: 'softmax'});
  const newOutputTensor = newDenseLayer.apply(cutoffTensor);

  const transferModel = tf.model({inputs: model.inputs, outputs: newOutputTensor});
  transferModel.compile({loss: 'categoricalCrossentropy',  optimizer: 'adam'});

  const numEpochs = 40;
  const plotData = {
    x: [],
    y: [],
    type: 'scatter',
  };
  const plotLayout = {
    xaxis: {range: [0, numEpochs], title: 'Epoch #'},
    yaxis: {title: 'Train loss'},
  };
  const history = await transferModel.fit(xs, ys, {
    epochs: numEpochs,
    callbacks: {
      onEpochEnd: async (epoch, log) => {
        plotData.x.push(epoch + 1);
        plotData.y.push(log.loss);
        Plotly.newPlot(transferLearnHistoryDiv, [plotData], plotLayout);
        await tf.nextFrame();
        console.log(`epoch = ${epoch}: loss = ${log.loss}`);
      }
    }
  });

  model = tf.model({
    inputs: model.inputs,
    outputs: model.outputs.concat(transferModel.outputs),
  });

  // TODO(cais): Save transfer words in localstorage.

  return history;
}

// Logic related to saving, loading and deleting local model.
const LOCAL_MODEL_SAVE_LOCATION = 'indexeddb://local-audio-model';
const MODEL_METADATA_SAVE_LOCATION = 'audio-model-metadata';
const TRANSFER_WORDS_SAVE_LOCATION = 'audio-model-transfer-words';

const loadModelFromLocalButton = document.getElementById('load-model-from-local');
const saveModelToLocalButton = document.getElementById('save-model-to-local');
const deleteModelFromLocalButton = document.getElementById('delete-model-from-local');

saveModelToLocalButton.addEventListener('click', async () => {
  // Save metadata: original words and frame size.
  const metadata = {
    frameSize: runOptions.frameSize,
    words: words
  };
  localStorage.setItem(
      MODEL_METADATA_SAVE_LOCATION, JSON.stringify(metadata));

  if (model.outputs.length > 1) {
    // Save transfer words.
    localStorage.setItem(
        TRANSFER_WORDS_SAVE_LOCATION, JSON.stringify(transferWords));
  }

  const saveResult = await model.save(LOCAL_MODEL_SAVE_LOCATION);
  logToStatusDisplay(
      `Saved model with ${model.outputs.length} output(s) to ` +
      `${LOCAL_MODEL_SAVE_LOCATION} at ` +
      `${saveResult.modelArtifactsInfo.dateSaved}`);
});

async function refreshLocalModelStatus() {
  const modelsInfo = await tf.io.listModels();
  if (modelsInfo[LOCAL_MODEL_SAVE_LOCATION] == null) {
    logToStatusDisplay('No locally-saved model is found.');
    loadModelFromLocalButton.disabled = true;
    deleteModelFromLocalButton.disabled = true;
  } else {
    logToStatusDisplay(
      `Locally-saved model available: saved at ` +
      `${modelsInfo[LOCAL_MODEL_SAVE_LOCATION].dateSaved}`);
    loadModelFromLocalButton.disabled = false;
    deleteModelFromLocalButton.disabled = false;
  }
}

refreshLocalModelStatus();

loadModelFromLocalButton.addEventListener('click', () => {
  loadModelAndMetadataAndWarmUpModel(false);
  saveModelToLocalButton.disabled = false;
  startButton.disabled = false;
});

deleteModelFromLocalButton.addEventListener('click', async () => {
  await tf.io.removeModel(LOCAL_MODEL_SAVE_LOCATION);
  localStorage.removeItem(MODEL_METADATA_SAVE_LOCATION);
  localStorage.removeItem(TRANSFER_WORDS_SAVE_LOCATION);
  await refreshLocalModelStatus();
});
