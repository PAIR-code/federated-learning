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

import * as tf from '@tensorflow/tfjs';
import MediaStreamRecorder from 'msr';
import {ClientAPI} from 'federated-learning-client';
import {plotSpectrogram, plotSpectrum} from './spectral_plots';
import {loadAudioTransferLearningModel} from './model';
import {FrequencyListener} from './frequency_listener';
import {getNextLabel, labelNames} from './labels';

const spectrumCanvas = document.getElementById('spectrum-canvas');
const modelDiv = document.getElementById('model');
const recordButton = document.getElementById('record-button');
const modelVersion = document.getElementById('model-version');
const labeledExamples = document.getElementById('labeled-examples');
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

let serverURL = location.origin;
if (URLSearchParams) {
  const params = new URLSearchParams(location.search);
  if (params.get('server')) {
    serverURL = params.get('server');
  }
}

loadAudioTransferLearningModel().then(async (model) => {
  const clientAPI = new ClientAPI(model);
  clientAPI.onDownload((msg) => {
    console.log(`new model: ${modelVersion.innerText} -> ${msg.modelVersion}`);
    modelVersion.innerText = msg.modelVersion;
  });
  await clientAPI.connect(serverURL);

  recordButton.innerHTML = 'Waiting for microphone&hellip;';
  const stream =
      await navigator.mediaDevices.getUserMedia({audio: true, video: false});

  setupUI(stream, model, clientAPI);
});

function setupUI(stream, model, clientAPI) {
  const inputShape = model.inputs[0].shape;
  const numFrames = inputShape[1];
  const fftLength = inputShape[2];
  let resultRow;
  let yTrue = getNextLabel();

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
    resultRow.children[0].appendChild(audioControls);

    // send .wav file to server
    const file =
        new File([blob], `${labelNames[yTrue]}.wav`, {type: 'audio/wav'});
    const req = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('file', file);
    req.open('POST', serverURL + '/data');
    req.send(formData);
  };

  // On each audio frame, update our rotating buffer with the latest FFT data
  // from our analyser. For fun, also update the spectrum plot
  const listener = FrequencyListener(stream, numFrames, fftLength);
  listener.onEachFrame(freqData => {
    plotSpectrum(spectrumCanvas, freqData, fftLength);
  });
  listener.listen();

  // Create our record button
  recordButton.innerHTML = 'Record';
  recordButton.removeAttribute('disabled');

  // When we click it, record for 1 second
  recordButton.addEventListener('click', () => {
    recordButton.innerHTML = 'Saving&hellip;';
    recordButton.setAttribute('disabled', 'disabled');
    modelDiv.innerHTML = waitingTemplate;
    recordButton.innerHTML = 'Listening&hellip;';
    recorder.start(1100);
    setTimeout(finishRecording, 1000);
  });

  // When we're done recording,
  function finishRecording() {
    // Setup results html
    modelDiv.innerHTML = modelTemplate;
    resultRow = document.createElement('tr');
    for (let i = 0; i < 4; i++)
      resultRow.appendChild(document.createElement('td'));
    labeledExamples.appendChild(resultRow);
    resultRow.children[2].innerText = labelNames[yTrue];

    // Stop separate .wav recording
    recorder.stop();

    // Compute input tensor
    const freqData = listener.getFrequencyData();
    const x = getInputTensorFromFrequencyData(freqData, numFrames, fftLength);

    // Plot spectrograms
    const mainSpectrogram = document.getElementById('spectrogram-canvas');
    const miniSpectrogram = document.createElement('canvas');
    miniSpectrogram.setAttribute('width', '81');
    miniSpectrogram.setAttribute('height', '54');
    plotSpectrogram(mainSpectrogram, freqData, fftLength);
    plotSpectrogram(miniSpectrogram, freqData, fftLength);
    resultRow.children[1].appendChild(miniSpectrogram);

    // Compute label tensor
    const onehotY = new Array(labelNames.length).fill(0);
    onehotY[yTrue] = 1;
    const y = tf.tensor2d([onehotY]);

    // Compute predictions
    let yPred;
    let probs;
    const p = model.predict(x);
    tf.tidy(() => {
      yPred = tf.argMax(p, 1).dataSync()[0];
      probs = p.dataSync();
    });
    resultRow.children[3].innerText = labelNames[yPred];
    if (yTrue != yPred) {
      resultRow.children[3].className = 'incorrect-prediction';
    }

    // Plot probabilities
    Plotly.newPlot('probs', [{x: labelNames, y: probs, type: 'bar'}], {
      autosize: false,
      width: 480,
      height: 180,
      margin: {l: 30, r: 5, b: 30, t: 5, pad: 0},
    });

    // Prepare our next loop...
    const cleanup = (err) => {
      if (err) {
        console.log('error uploading data or fitting model:');
        console.log(err);
      }
      // dispose of tensors
      tf.dispose([x, y, p]);

      // decide what label to request next
      yTrue = getNextLabel();

      // re-allow recording
      recordButton.innerText = 'Record';
      recordButton.removeAttribute('disabled');
      console.log('...done!');
    };

    // ...after we upload data and train
    console.log('uploading data...');
    recordButton.innerHTML = 'Uploading Data&hellip;'
    clientAPI.uploadData(x, y, p, labelNames[yTrue], labelNames[yPred]).then(() => {
      console.log('fitting model...');
      recordButton.innerHTML = 'Fitting Model&hellip;'
      clientAPI.federatedUpdate(x, y).then(cleanup, cleanup);
    }, cleanup);
  }
}

function normalize(x) {
  return tf.tidy(() => {
    const mean = tf.mean(x);
    const std = tf.sqrt(tf.mean(tf.square(tf.add(x, tf.neg(mean)))));
    return tf.div(tf.add(x, tf.neg(mean)), std);
  });
}

function getInputTensorFromFrequencyData(freqData, numFrames, fftLength) {
  return tf.tidy(() => {
    const size = freqData.length;
    const tensorBuffer = tf.buffer([size]);
    for (let i = 0; i < freqData.length; ++i) {
      tensorBuffer.set(freqData[i], i);
    }
    return normalize(tensorBuffer.toTensor().reshape(
        [1, numFrames, fftLength, 1]));
  });
}
