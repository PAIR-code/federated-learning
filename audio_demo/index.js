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
import MediaStreamRecorder from 'msr';
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
const labeledExamples = document.getElementById('labeled-examples');
const firstIntro = "Would you be willing to help me?"
  + " I'd love it if you could show me how to pronounce the word:";
const laterIntro = "If you're up for another, " +
  "could you show me how to pronounce:";
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
    <label>Input</label>
    <canvas id="spectrogram-canvas" height="180" width="270"></canvas>
  </div>
  <div class='chart'>
    <label>Output</label>
    <div id='probs'></div>
  </div>
`;
const serverURL = location.href.replace('1234', '3000');
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
    console.log(
      `new model! updating from ${modelVersion.innerText} to ${msg.modelId}`);
    modelVersion.innerText = msg.modelId;
    return true;
  }
  window.clientAPI = clientAPI;
  clientAPI.initialise(serverURL).then((fitConfig) => {
    modelVersion.innerText = clientAPI.modelId;
    window.fitConfig = fitConfig;
    recordButton.innerHTML = 'Waiting for microphone&hellip;';
    navigator.mediaDevices.getUserMedia({audio: true, video: false})
      .then(stream => setupUI(stream));
  });
});

function setupUI(stream) {
  // Ask user to provide audio for all labels, but in a random order
  const randomLabels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13];
  shuffle(randomLabels);

  // Set up audio objects that listen to our stream
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
  const frameDurationMillis =
    runOptions.frameSize / runOptions.sampleRate * 1e3;
  const waitingPeriodFrames = Math.round(
    runOptions.waitingPeriodMillis / frameDurationMillis);
  const refractoryPeriodFrames = Math.round(
    runOptions.refractoryPeriodMillis / frameDurationMillis);
  const tracker = new Tracker(waitingPeriodFrames, refractoryPeriodFrames);

  // Keep track of counters that change each audio frame or recording iteration
  let frameCount = 0;
  let labelIdx = 0;
  let thanksIdx = 0;
  suggestedLabel.innerText = labelNames[randomLabels[labelIdx]];

  // Create a recorder to save the raw .wav file
  const recorder = new MediaStreamRecorder(stream);
  recorder.mimeType = 'audio/wav';
  recorder.ondataavailable = blob => {
    // create audio element with recording
    const url = URL.createObjectURL(blob);
    const audioControls = document.createElement('audio');
    const audioSource = document.createElement('source');
    audioControls.setAttribute('controls', 'controls');
    audioSource.src = url;
    audioSource.type = 'audio/wav';
    audioControls.appendChild(audioSource);
    tr.children[0].appendChild(audioControls);

    // send .wav file to server
    const file = new File([blob], `${labelNames[yTrue]}.wav`, {
      type: 'audio/wav'
    });
    const req = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('file', file);
    req.open('POST', serverURL + 'data');
    req.send(formData);
  };

  // On each audio frame, update our rotating buffer with the latest FFT data
  // from our analyser. For fun, also update the spectrum plot
  function handleAudioFrame() {
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
  setInterval(handleAudioFrame,
    analyser.frequencyBinCount / audioContext.sampleRate * 1000);

  // Create our record button
  introText.innerText = firstIntro;
  recordButton.innerHTML = 'Record';
  recordButton.removeAttribute('disabled');

  // When we click it, record for 1 second
  recordButton.addEventListener('click', async (event) => {
    recordButton.innerHTML = 'Saving&hellip;';
    recordButton.setAttribute('disabled', 'disabled');
    modelDiv.innerHTML = waitingTemplate;
    recordButton.innerHTML = "Listening&hellip;";
    recorder.start(1100);
    setTimeout(finishRecording, 1000);
  });

  // When we're done recording,
  function finishRecording() {
    // Setup results html
    modelDiv.innerHTML = modelTemplate;
    window.tr = document.createElement('tr');
    for (let i = 0; i < 4; i++)
      tr.appendChild(document.createElement('td'));
    labeledExamples.appendChild(tr);

    // Compute true prediction
    const yTrue = randomLabels[labelIdx];
    window.yTrue = yTrue;
    tr.children[2].innerText = labelNames[yTrue];

    // Stop separate .wav recording
    recorder.stop();

    // Compute input tensor
    const freqData = getFrequencyDataFromRotatingBuffer(
      rotatingBuffer, frameCount - runOptions.numFrames);
    const x = getInputTensorFromFrequencyData(freqData);

    // Plot spectrograms
    const mainSpectrogram = document.getElementById('spectrogram-canvas');
    const miniSpectrogram = document.createElement('canvas');
    miniSpectrogram.setAttribute('width', '81');
    miniSpectrogram.setAttribute('height', '54');
    plotSpectrogram(
      mainSpectrogram, freqData,
      runOptions.modelFFTLength, runOptions.modelFFTLength);
    plotSpectrogram(
      miniSpectrogram, freqData,
      runOptions.modelFFTLength, runOptions.modelFFTLength);
    tr.children[1].appendChild(miniSpectrogram);

    // Compute label tensor
    const y = toOneHot(yTrue);
    const ys = tf.expandDims(y);

    // Compute predictions
    let probs;
    tf.tidy(() => {
      probs = model.predict(x).dataSync();
    });
    const yPred = getArgMax(probs);
    tr.children[3].innerText = labelNames[yPred];
    if (yTrue != yPred) {
      tr.children[3].className = 'incorrect-prediction';
    }

    // Plot probabilities
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

    // Prepare our next loop...
    const cleanup = (err) => {
      if (err) {
        console.log('error uploading data or fitting model:');
        console.log(err);
      }
      // restore variables we had before training
      clientAPI.revertToOriginalVars();
      clientAPI.numExamples = 0;
      // dispose of tensors
      ys.dispose();
      x.dispose();
      y.dispose();
      console.log(tf.memory());
      // decide what label to request next
      labelIdx += 1;
      if (labelIdx >= labelNames.length) {
        labelIdx = 0;
        shuffle(randomLabels); // reshuffle each iteration
      }
      suggestedLabel.innerText = labelNames[randomLabels[labelIdx]];
      // thank the user
      introText.innerText = thanksVariants[thanksIdx] + ' ' + laterIntro;
      thanksIdx = (thanksIdx + 1) % thanksVariants.length;
      // re-allow recording
      recordButton.innerText = 'Record';
      recordButton.removeAttribute('disabled');
    };

    // ...after we upload data and train
    console.log('uploading data...');
    recordButton.innerHTML = 'Uploading Data&hellip;'
    clientAPI.uploadData(x, y).then(() => {
      console.log('fitting model...');
      const modelVersionBeforeFitting = clientAPI.modelId;
      recordButton.innerHTML = 'Fitting Model&hellip;'
      model.fit(x, ys).then(() => {
        if (clientAPI.modelId === modelVersionBeforeFitting) {
          console.log('uploading weights...');
          clientAPI.numExamples = 1;
          recordButton.innerHTML = 'Uploading Weights&hellip;'
          clientAPI.uploadVars().then(cleanup, cleanup);
        } else {
          console.log('aborting weight upload due to version change!');
          cleanup();
        }
      }, cleanup);
    }, cleanup);
  }
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
