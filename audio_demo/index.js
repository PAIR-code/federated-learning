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

const labelNames =
    'one,two,three,four,five,six,seven,eight,nine,zero,left,right,go,stop'.split(',');

const audioTransferLearningModelURL =
    'https://storage.googleapis.com/tfjs-speech-command-model-14w/model.json';

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
const modelOutputDiv = document.getElementById('model-output');
const modelInputDiv = document.getElementById('model-input');
const recordButton = document.getElementById('record-button');
const modelVersion = document.getElementById('model-version');
let recording = false;

const waitingTemplate = `Waiting&hellip;`;

const inputTemplate = `
  <div class='chart'>
    <label>Spectrogram</label>
    <canvas id="spectrogram-canvas" height="180" width="270"></canvas>
  </div>
  <div class='chart'>
    <label>Recording</label>
    <audio controls id='audio-controls'></audio>
  </div>
`;

let outputTemplate = `
  <label>Prediction: <em id='predicted-label'>&hellip;</em></label>
  <div id='probs'></div>
  <br>
  <label>What <em>should</em> we have chosen?</label>
  <ul id='label-radios'>
`;
for (let i = 0; i < labelNames.length; i++) {
  outputTemplate += `<li><label>${labelNames[i]}<input type='radio' name='true-label' value='${i}'></label></li>`;
}
outputTemplate += `<li><label>Nothing (ignore this one)<input type='radio' name='true-label' value=''></label></li>`;
outputTemplate += "</ul>";

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
  let audioTask;
  let webmChunks;

  const recorder = new MediaRecorder(stream);
  recorder.ondataavailable = event => {
    webmChunks.push(event.data);
    if (recorder.state == 'inactive') {
      const audioBlob = new Blob(webmChunks, { type: 'audio/webm' });
      const blobUrl = URL.createObjectURL(audioBlob);
      const audioControls = document.getElementById('audio-controls');
      audioControls.innerHTML = '';
      const sourceEl = document.createElement('source');
      sourceEl.src = blobUrl;
      sourceEl.type = 'audio/webm';
      audioControls.appendChild(sourceEl);
    }
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

    if (!recording) {
      return;
    }

    const spectralMax = getArrayMax(freqDataSlice);
    tracker.tick(spectralMax > runOptions.magnitudeThreshold);
    if (tracker.shouldFire()) {
      modelOutputDiv.innerHTML = outputTemplate;
      modelInputDiv.innerHTML = inputTemplate;
      recording = false;
      recorder.stop();
      recordButton.innerText = 'Save & Record New';
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
          width: 600,
          height: 200,
          margin: { l: 30, r: 5, b: 20, t: 5, pad: 0 },
        });

        document.getElementById('predicted-label').innerText = labelNames[getArgMax(probs)];
      });
    }
    frameCount++;
  }

  window.inputTensors = [];

  const frameFreq = analyser.frequencyBinCount / audioContext.sampleRate * 1000;
  audioTask = setInterval(onEveryAudioFrame, frameFreq);
  recordButton.innerHTML = 'Record Sample';
  recordButton.removeAttribute('disabled');

  recordButton.addEventListener('click', async (event) => {
    recordButton.innerHTML = 'Saving&hellip;';
    recordButton.setAttribute('disabled', 'disabled');
    await saveLabeledExample();
    modelOutputDiv.innerHTML = waitingTemplate;
    modelInputDiv.innerHTML = waitingTemplate;
    recording = true;
    webmChunks = [];
    clearInterval(audioTask);
    audioTask = setInterval(onEveryAudioFrame, frameFreq);
    recorder.start(10);
    recordButton.innerHTML = "Listening&hellip;";
  });
}

tf.loadModel(audioTransferLearningModelURL).then((model) => {
  const inputShape = model.inputs[0].shape;
  runOptions.numFrames = inputShape[1];
  runOptions.modelFFTLength = inputShape[2];
  runOptions.frameMillis = runOptions.frameSize / runOptions.sampleRate * 1e3;
  window.model = model;
  for (let i = 0; i < 9; ++i) {
    model.layers[i].trainable = false;  // freeze conv layers
  }
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
})

async function saveLabeledExample() {
  const x = window.inputTensors[window.inputTensors.length-1];
  const enteredLabel = document.querySelector('input[name="true-label"]:checked');
  if (enteredLabel && enteredLabel.value != '') {
    const yTrue = parseInt(enteredLabel.value);
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
      console.log('upload data');
      await window.clientAPI.uploadData(x, y);

      console.log('train model');
      await model.fit(x, ys, fitConfig);

      console.log('upload vars');
      clientAPI.numExamples = 1;
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
  } else if (x) {
    x.dispose();
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
