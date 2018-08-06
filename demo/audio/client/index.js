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
import * as npy from './npy';
import uuid from 'uuid/v4';
import MediaStreamRecorder from 'msr';
import {ClientAPI} from 'federated-learning-client';
import {plotSpectrogram, plotSpectrum} from './spectral_plots';
import {loadAudioTransferLearningModel} from './model';
import {FrequencyListener} from './frequency_listener';
import {labelNames} from './labels';

window.tf = tf;

const recordFieldset = document.getElementById('spells');
labelNames.forEach(name => {
  const button = document.createElement('button');
  button.classList.add('record-button');
  button.setAttribute('value', name);
  button.setAttribute('disabled', 'disabled');
  button.innerText = name;
  recordFieldset.appendChild(button);
});
const lastPrediction = document.createElement('div');
recordFieldset.appendChild(lastPrediction);

const htmlEl = document.getElementsByTagName('html')[0];
const spectrumCanvas = document.getElementById('spectrum-canvas');
const modelDiv = document.getElementById('model');
const modelVersion = document.getElementById('model-version');
const statusBar = document.getElementById('status');
const recordButtons = Array.from(
    document.getElementsByClassName('record-button'));
const labeledExamples = document.getElementById('labeled-examples');
const waitingTemplate = `Waiting for input&hellip;`;
const modelTemplate = `
  <div class='chart model-input'>
    <label>Input</label>
    <canvas id="spectrogram-canvas" height="180" width="270"></canvas>
  </div>
  <div class='chart model-output'>
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

function setStatus(statusHTML) {
  statusBar.innerHTML = statusHTML;
}

function setReadyStatus(clientAPI) {
  let html = 'Ready to listen!';
  const need = clientAPI.numExamplesPerUpdate() - clientAPI.numExamples();
  html += ` Need <span class='status-number'>${need}</span> more before training.`
  const trained = clientAPI.numUpdates();
  if (trained) {
    html += ` You've trained <span class='status-number'>${trained}</span> 🧙`;
  }
  setStatus(html);
}

loadAudioTransferLearningModel().then(async (model) => {
  const clientAPI = new ClientAPI(serverURL, model);

  clientAPI.onNewVersion((model, oldVersion, newVersion) => {
    console.log(`${oldVersion} -> ${newVersion}`);
    modelVersion.innerText = `version #${clientAPI.numVersions()}`;
  });

  const t1 = new Date().getTime();
  await clientAPI.setup();
  const t2 = new Date().getTime();
  console.log(`download took ${t2-t1}ms`);

  window.api = clientAPI;
  window.model = clientAPI.model;

  setStatus('Waiting for microphone&hellip;');
  const stream =
      await navigator.mediaDevices.getUserMedia({audio: true, video: false});

  setupUI(stream, model, clientAPI);
});

function setupUI(stream, model, clientAPI) {
  const inputShape = model.inputs[0].shape;
  const numFrames = inputShape[1];
  const fftLength = inputShape[2];
  let resultRow;
  let yTrue;// = getNextLabel();
  let yPred;
  let probs;
  let xNpy;
  let metrics;

  if (!getCookie('federated-learner-uuid')) {
    setCookie('federated-learner-uuid', uuid());
  }

  const clientId = getCookie('federated-learner-uuid');

  // Create a recorder to save the raw .wav file
  const recorder = new MediaStreamRecorder(stream);
  recorder.mimeType = 'audio/wav';
  recorder.ondataavailable = wavBlob => {
    // create audio element with recording
    const url = URL.createObjectURL(wavBlob);
    const audioControls = document.createElement('audio');
    const audioSource = document.createElement('source');
    audioControls.setAttribute('controls', 'controls');
    audioSource.src = url;
    audioSource.type = 'audio/wav';
    audioControls.appendChild(audioSource);
    resultRow.children[0].appendChild(audioControls);

    const fn = labelNames[yTrue];
    const npyBlob = new Blob([new Uint8Array(xNpy)]);
    const wavFile =
        new File([wavBlob], `${fn}.wav`, {type: 'audio/wav'});
    const npyFile =
        new File([npyBlob], `${fn}.npy`, {type: 'application/octet-stream'});
    // send .wav file to server
    const req = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('wav', wavFile);
    formData.append('npy', npyFile);
    formData.append('clientId', clientId);
    formData.append('modelVersion', clientAPI.modelVersion());
    formData.append('timestamp', new Date().getTime().toString());
    formData.append('predictedLabel', yPred);
    formData.append('trueLabel', yTrue);
    formData.append('modelOutput', probs);
    formData.append('metrics', JSON.stringify(metrics));
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
  setReadyStatus(clientAPI);
  recordButtons.forEach(b => b.removeAttribute('disabled'));

  // When we click it, record for 1 second
  recordButtons.forEach(button => {
    button.addEventListener('click', () => {
      lastPrediction.innerHTML = '';
      recordButtons.forEach(b => b.setAttribute('disabled', 'disabled'));
      yTrue = labelNames.indexOf(button.getAttribute('value'));
      button.classList.add('active');
      console.log(yTrue);
      modelDiv.innerHTML = waitingTemplate;
      setStatus('Listening&hellip;');
      recorder.start(1100);
      setTimeout(finishRecording, 1000);
    });
  });

  // When we're done recording,
  async function finishRecording() {
    // Setup results html
    modelDiv.innerHTML = modelTemplate;
    resultRow = document.createElement('tr');
    for (let i = 0; i < 4; i++)
      resultRow.appendChild(document.createElement('td'));
    labeledExamples.prepend(resultRow);
    resultRow.children[2].innerText = labelNames[yTrue];

    // Compute input tensor
    const freqData = listener.getFrequencyData();
    const x = getInputTensorFromFrequencyData(freqData, numFrames, fftLength);
    xNpy = npy.serialize(x);

    // Compute label tensor
    const onehotY = new Array(labelNames.length).fill(0);
    onehotY[yTrue] = 1;
    const y = tf.tensor2d([onehotY]);

    // Compute predictions
    const p = model.predict(x);
    tf.tidy(() => {
      yPred = tf.argMax(p, 1).dataSync()[0];
      probs = p.dataSync();
    });
    metrics = await clientAPI.model.evaluate(x, y);

    // Stop separate .wav recording
    recorder.stop();

    // Plot spectrograms
    const mainSpectrogram = document.getElementById('spectrogram-canvas');
    const miniSpectrogram = document.createElement('canvas');
    miniSpectrogram.setAttribute('width', '81');
    miniSpectrogram.setAttribute('height', '54');
    plotSpectrogram(mainSpectrogram, freqData, fftLength);
    plotSpectrogram(miniSpectrogram, freqData, fftLength);
    resultRow.children[1].appendChild(miniSpectrogram);

    resultRow.children[3].innerText = labelNames[yPred];
    if (yTrue != yPred) {
      resultRow.children[3].className = 'incorrect-prediction';
      lastPrediction.innerHTML = `You tried to cast <span class='last-prediction'>${labelNames[yTrue]}</span>, but your wand performed <span class='last-prediction'>${labelNames[yPred]}</span>!`;
    } else {
      lastPrediction.innerHTML = `You successfully cast <span class='last-prediction'>${labelNames[yPred]}</span>!`;
    }

    switch (labelNames[yPred]) {
      case 'nox':
        htmlEl.classList.add('nox');
        break;
      case 'lumos':
        htmlEl.classList.remove('nox');
        break;
      case 'accio':
        htmlEl.classList.add('accio');
        break;
      case 'expelliarmus':
        htmlEl.classList.remove('accio');
    }


    // Plot probabilities
    Plotly.newPlot('probs', [{x: labelNames, y: probs, type: 'bar'}], {
      autosize: false,
      width: Math.min(270*1.25, document.getElementById('model').clientWidth),
      height: 180,
      margin: {l: 30, r: 5, b: 30, t: 5, pad: 0},
    });

    // Prepare our next loop...
    const cleanup = (err) => {
      // dispose of tensors
      tf.dispose([x, y, p]);

      // re-allow recording
      recordButtons.forEach(b => {
        b.removeAttribute('disabled');
        b.classList.remove('active');
      });

      setReadyStatus(clientAPI);
      console.log('...done!');
    };

    // ...after we train
    console.log('fitting model...');
    setStatus('Fitting Model&hellip;')
    clientAPI.federatedUpdate(x, y).catch(console.log).finally(cleanup);
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

function getCookie(name) {
  var v = document.cookie.match('(^|;) ?' + name + '=([^;]*)(;|$)');
  return v ? v[2] : null;
}

function setCookie(name, value) {
  var d = new Date;
  d.setTime(d.getTime() + 24*60*60*1000*365);
  document.cookie = name + "=" + value + ";path=/;expires=" + d.toGMTString();
}
