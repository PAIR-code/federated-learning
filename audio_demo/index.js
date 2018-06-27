/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as client from '../src/client/client';
import * as tf from '@tensorflow/tfjs';
import Tracker from './tracker';
import {plotSpectrogram, plotSpectrum} from './spectral_plots';
import {AudioTransferLearningModel} from '../src/index';

const labelNames =
    'one,two,three,four,five,six,seven,eight,nine,zero,left,right,go,stop'.split(',');

const runOptions = {
  magnitudeThreshold: -40,
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

const spectrumCanvas = document.getElementById('spectrum-canvas');
const modelDiv = document.getElementById('model');
const recordButton = document.getElementById('record-button');
const modelVersion = document.getElementById('model-version');
const suggestedLabel = document.getElementById('suggested-label');
const introText = document.getElementById('intro-text');
let recording = false;

const firstIntro = "Would you be willing to help me? I'd love it if you could show me how to pronounce the word:"
const laterIntro = "If you're up for another, could you show me how to pronounce:"
const thanksVariants = [
  "Thanks!", "Gracias!", "Much obliged!", "Bravo!",
  "Thanks!", "Gracias!", "Much obliged!", "Bravo!",
  "Not bad!",
  "Thanks!", "Gracias!", "Much obliged!", "Bravo!",
  "You're getting good at this!",
  "Thanks!", "Gracias!", "Much obliged!", "Bravo!",
  "Thanks!", "Gracias!", "Much obliged!", "Bravo!",
  "No one expects the Spanish Inquisition!"
];
const waitingTemplate = `Waiting for input&hellip;`;
const modelTemplate = `
  <div class='chart'>
    <label>Spectrogram</label>
    <canvas id="spectrogram-canvas" height="180" width="270"></canvas>
  </div>
  <div class='chart'>
    <label>Prediction</label>
    <div id='probs'></div>
  </div>
  <div class='chart'>
    <label>Recording</label>
    <audio controls id='audio-controls'></audio>
  </div>
`;

function setupUI(stream) {
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioContext();
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
  const tracker = new Tracker(waitingPeriodFrames, refractoryPeriodFrames);
  const randomLabels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13];
  shuffle(randomLabels);
  let labelIdx = 0;
  let thanksIdx = 0;
  suggestedLabel.innerText = labelNames[randomLabels[labelIdx]];

  const recorder = new MediaStreamRecorder(stream, { })
  recorder.mimeType = 'audio/wav';
  recorder.ondataavailable = blob => {
    const url = URL.createObjectURL(blob);
    console.log(url);
    const audioControls = document.getElementById('audio-controls');
    audioControls.innerHTML = '';
    const sourceEl = document.createElement('source');
    sourceEl.src = url;
    sourceEl.type = 'audio/wav';
    audioControls.appendChild(sourceEl);
  };

  function onEveryAudioFrame() {
    analyser.getFloatFrequencyData(freqData);
    if (freqData[0] === -Infinity && freqData[1] === -Infinity) {
      return;
    }
    const freqDataSlice = freqData.slice(0, runOptions.modelFFTLength);
    plotSpectrum(spectrumCanvas, freqDataSlice, runOptions);
    const bufferPos = frameCount % rotatingBufferNumFrames;
    rotatingBuffer.set(freqDataSlice, bufferPos * runOptions.modelFFTLength);
    frameCount++;
  }

  window.inputTensors = [];

  function stopRecording() {
    recording = false;
    modelDiv.innerHTML = modelTemplate;
    recorder.stop();
    recordButton.innerText = 'Record';
    recordButton.removeAttribute('disabled');
    const freqData = getFrequencyDataFromRotatingBuffer(
      rotatingBuffer, frameCount - runOptions.numFrames);
    const spectrogramCanvas = document.getElementById('spectrogram-canvas');
    plotSpectrogram(
      spectrogramCanvas, freqData,
      runOptions.modelFFTLength, runOptions.modelFFTLength);
    const inputTensor = getInputTensorFromFrequencyData(freqData);
    window.inputTensors.push(inputTensor);
    tf.tidy(() => {
      const probs = model.predict(inputTensor).dataSync();
      window.probs = probs;
      Plotly.newPlot('probs', [{
        x: labelNames,
        y: probs,
        type: 'bar'
      }], {
        autosize: false,
        width: 480,
        height: 180,
        margin: { l: 30, r: 5, b: 30, t: 5, pad: 0 },
      });
    });
    saveLabeledExample(inputTensor, randomLabels[labelIdx]);
    labelIdx = (labelIdx + 1) % labelNames.length;
    suggestedLabel.innerText = labelNames[randomLabels[labelIdx]];
    introText.innerText = thanksVariants[thanksIdx] + ' ' + laterIntro;
    thanksIdx = (thanksIdx + 1) % thanksVariants.length;
  }
  const frameFreq = analyser.frequencyBinCount / audioContext.sampleRate * 1000;
  setInterval(onEveryAudioFrame, frameFreq);

  introText.innerText = firstIntro;
  recordButton.innerHTML = 'Record';
  recordButton.removeAttribute('disabled');
  recordButton.addEventListener('click', async (event) => {
    recordButton.innerHTML = 'Saving&hellip;';
    recordButton.setAttribute('disabled', 'disabled');
    modelDiv.innerHTML = waitingTemplate;
    recordButton.innerHTML = "Listening&hellip;";
    recording = true;
    recorder.start(1100);
    setTimeout(stopRecording, 1000);
  });
}

const fedModel = new AudioTransferLearningModel();
fedModel.setup().then((dict) => {
  const model = dict.model;
  const inputShape = model.inputs[0].shape;
  runOptions.numFrames = inputShape[1];
  runOptions.modelFFTLength = inputShape[2];
  runOptions.frameMillis = runOptions.frameSize / runOptions.sampleRate * 1e3;
  window.model = model;
  model.compile({'optimizer': 'sgd', loss: 'categoricalCrossentropy'});
  const clientAPI = client.VariableSynchroniser.fromLayers(model.layers);
  clientAPI.acceptUpdate = (msg) => {
    modelVersion.innerText = msg.modelId;
    return true;
  }
  window.clientAPI = clientAPI;
  clientAPI.initialise(location.href.replace('1234', '3000')).then((fitConfig) => {
    modelVersion.innerText = clientAPI.modelId;
    window.fitConfig = fitConfig;
    recordButton.innerHTML = 'Waiting for microphone&hellip;';
    navigator.mediaDevices.getUserMedia({audio: true, video: false})
      .then(stream => setupUI(stream));
  });
});

async function saveLabeledExample(x, yTrue) {
  const yPred = getArgMax(window.probs);
  const spectrogramCanvas = document.getElementById('spectrogram-canvas');
  spectrogramCanvas.parentNode.removeChild(spectrogramCanvas);
  spectrogramCanvas.setAttribute('id', '');
  spectrogramCanvas.setAttribute('style', 'width: 54px; height: 32px;');
  const tr = document.createElement('tr');
  const td1 = document.createElement('td');
  const td2 = document.createElement('td');
  const td3 = document.createElement('td');
  const td4 = document.createElement('td');
  td1.appendChild(spectrogramCanvas);
  td2.innerText = labelNames[yTrue];
  td3.innerText = labelNames[yPred];
  if (yTrue != yPred) {
    td3.className = 'incorrect-prediction';
  }
  td4.innerHTML = '&hellip;';
  tr.appendChild(td1);
  tr.appendChild(td2);
  tr.appendChild(td3);
  tr.appendChild(td4);
  document.getElementById('labeled-examples').appendChild(tr);
  window.probs = undefined;
  const y = toOneHot(yTrue);
  const ys = tf.expandDims(y);
  try {
    clientAPI.numExamples = 1;
    await window.clientAPI.uploadData(x, y);
    await model.fit(x, ys, fitConfig);
    await clientAPI.uploadVars();
    td4.innerText = '✔️';
  } catch (error) {
    console.log(error);
    td4.innerText = '✗';
  } finally {
    clientAPI.revertToOriginalVars();
    clientAPI.numExamples = 0;
    ys.dispose();
    x.dispose();
    y.dispose();
  }
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

function getArgMax(xs) {
  let max = -Infinity;
  let idx = 0;
  for (let i = 0; i < xs.length; ++i) {
    if (xs[i] > max) {
      max = xs[i];
      idx = i;
    }
  }
  return idx;
}

function toOneHot(j) {
  const yArr = [];
  for (let i = 0; i < labelNames.length; i++) {
    yArr.push(0);
  }
  yArr[j] = 1;
  return tf.tensor1d(yArr);
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

function normalize(x) {
  console.log('is x a tensor? ' + client.isTensor(x));
  console.log('is x a tensor? ' + (x instanceof tf.Tensor));
  const mean = tf.mean(x);
  const std = tf.sqrt(tf.mean(tf.square(tf.add(x, tf.neg(mean)))));
  return tf.div(tf.add(x, tf.neg(mean)), std);
}

function getInputTensorFromFrequencyData(freqData) {
  return tf.tidy(() => {
    const size = freqData.length;
    const tensorBuffer = tf.buffer([size]);
    for (let i = 0; i < freqData.length; ++i) {
      tensorBuffer.set(freqData[i], i);
    }
    return normalize(tensorBuffer.toTensor().reshape([
    1, runOptions.numFrames, runOptions.modelFFTLength, 1]));
  });
}

function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
