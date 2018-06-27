const mainCanvas = document.getElementById('main-canvas');

const datFileInput = document.getElementById('dat-file-input');
const startDatFileButton = document.getElementById('start-dat-file');
const datProgress = document.getElementById('dat-progress');
const showSpectrogram = document.getElementById('show-spectrogram');

const modelURLInput = document.getElementById('model-url');
const loadModelButton = document.getElementById('load-model');
const labelsInput = document.getElementById('labels');
const predictionResult = document.getElementById('prediction-result');
const predictionWinner = document.getElementById('prediction-winner');

let numRecordings = -1;
let recordingCounter = 0;
let recordingNumFrames = [];

let numFrames = -1;

const samplingFrequency = 44100;
// 1.1 seconds gives us some comfortable wiggle room. Most of the recordings are
// about 1 second long.
const maxRecordingLengthSeconds = 1.1;

let frameSize = 1024;
let outputArrays = null;
let intervalTask = null;

// Loaded model that can be used to run prediction on conversion results.
let model;

function triggerDatFileDownload(outputArray) {
  const anchor = document.createElement('a');
  anchor.download = 'output.dat';
  anchor.href = URL.createObjectURL(new Blob(
      [outputArray], {type: 'application/octet-stream'}));
  anchor.click();
}

function triggerCombinedDatFileDownload(outputArrays) {
  const anchor = document.createElement('a');
  anchor.download = 'combined.dat';
  anchor.href = URL.createObjectURL(new Blob(
      outputArrays, {type: 'application/octet-stream'}));
  anchor.click();
}

function createBufferWithValues(audioContext, xs) {
  const bufferLen = xs.length;
  const buffer = audioContext.createBuffer(
      1, bufferLen, audioContext.sampleRate);
  const channelData = buffer.getChannelData(0);
  for (let i = 0; i < bufferLen; ++i) {
    channelData[i] = xs[i];
  }
  numFrames = Math.floor(buffer.length / frameSize) + 5;
  const arrayLength = frameSize * (numFrames + 1);
  outputArrays.push(createAllMinusInfinityFloat32Array(arrayLength));
  return buffer;
}

function createAllMinusInfinityFloat32Array(arrayLength) {

  const outputArray = new Float32Array(arrayLength);
  for (let i = 0; i < arrayLength; ++i) {
    outputArray[i] = -Infinity;
  }
  return outputArray;
}

function startNewRecording() {
  if (numRecordings > 0 && recordingCounter >= numRecordings) {
    console.log('Downloading combined data file...');
    triggerCombinedDatFileDownload(outputArrays);
    return;
  }

  const offlineContext = new OfflineAudioContext(
      1, samplingFrequency * maxRecordingLengthSeconds * 4, samplingFrequency);
  const reader = new FileReader();
  reader.onloadend = async () => {
    const dat = new Float32Array(reader.result);
    const source = offlineContext.createBufferSource();
    source.buffer = createBufferWithValues(offlineContext, dat);

    const analyser = offlineContext.createAnalyser();
    analyser.fftSize = frameSize * 2;
    analyser.smoothingTimeConstant = 0.0;
    const freqData = new Float32Array(analyser.frequencyBinCount);

    source.connect(analyser);
    analyser.connect(offlineContext.destination);
    source.start();

    let recordingConversionSucceeded = false;
    let frameCounter = 0;
    const frameDuration = frameSize / samplingFrequency;
    offlineContext.suspend(frameDuration).then(async () => {
      analyser.getFloatFrequencyData(freqData);
      const outputArray = outputArrays[outputArrays.length - 1];
      outputArray.set(freqData, frameCounter * analyser.frequencyBinCount);

      while (true) {
        frameCounter++;
        offlineContext.resume();
        try {
          await offlineContext.suspend((frameCounter + 1) * frameDuration);
        } catch(err) {
          console.log(
              `suspend() call failed. Retrying file #${recordingCounter}: ` +
              datFileInput.files[recordingCounter].name);
          break;
        }

        analyser.getFloatFrequencyData(freqData);
        if (freqData[0] === -Infinity && freqData[1] === -Infinity) {
          recordingConversionSucceeded = true;
          break;
        }
        const outputArray = outputArrays[outputArrays.length - 1];
        outputArray.set(freqData, frameCounter * analyser.frequencyBinCount);
      }

      if (recordingConversionSucceeded) {
        recordingCounter++;
        datProgress.textContent = `Converting #${recordingCounter}`;
        if (showSpectrogram.checked) {
          plotSpectrogram(mainCanvas, outputArray, frameSize, 256);
        }
        if (model != null) {
          plotSpectrogram(mainCanvas, outputArray, frameSize, 256);
          runPrediction(outputArray);
        }
        setTimeout(startNewRecording, 5);
      } else {
        outputArrays.pop();
        source.stop();
        setTimeout(startNewRecording, 20);
      }
    });
    offlineContext.startRendering().catch(err => {
      console.log('Failed to render offline audio context:', err);
    });
  };
  recordingFile = datFileInput.files[recordingCounter];

  reader.readAsArrayBuffer(recordingFile);
}

startDatFileButton.addEventListener('click', event => {
  if (datFileInput.files.length > 0) {
    outputArrays = [];
    numRecordings = datFileInput.files.length;
    recordingNumFrames = [];
    recordingCounter = 0;
    startNewRecording();
  } else {
    alert('Select one or more files first.');
  }
});

loadModelButton.addEventListener('click', async () => {
  console.log(modelURLInput.value);
  model = await tf.loadModel(modelURLInput.value);
  loadModelButton.disabled = true;
});

function runPrediction(dataArray) {
  if (model == null) {
    throw new Error('Model is not loaded yet');
  }
  const timeSteps = model.inputs[0].shape[1];
  const freqSteps = model.inputs[0].shape[2];
  const tensorBuffer = tf.buffer([timeSteps * freqSteps]);
  let k = 0;
  for (let i = 0; i < timeSteps; ++i) {
    for (let j = 0; j < freqSteps; ++j) {
      const x = dataArray[i * frameSize + j];
      tensorBuffer.set(x, k++);
    }
  }
  const unnormalized = tensorBuffer.toTensor();
  const normalized = normalize(unnormalized).reshape(
      [1, timeSteps, freqSteps, 1]);
  const predictOut = model.predict(normalized).dataSync();

  const labels = labelsInput.value.split(',');
  const word2Score = {};
  let maxScore = -Infinity;
  let winnerIndex = -1;
  predictOut.forEach((score, index) => {
    word2Score[labels[index]] = score.toFixed(3);
    if (score > maxScore) {
      maxScore = score;
      winnerIndex = index;
    }
  });
  predictionResult.value = JSON.stringify(word2Score);
  predictionWinner.value += labels[winnerIndex] + ',';
}
