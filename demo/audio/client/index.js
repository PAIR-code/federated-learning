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
import * as npy from 'tfjs-npy';
import MediaStreamRecorder from 'msr';
import {loadAudioTransferLearningModel, labelNames} from './model';
import {FrequencyListener} from './frequency_listener';
import * as federated from 'federated-learning-client';
import * as ui from './ui';

let serverURL = location.origin;
if (URLSearchParams) {
  const params = new URLSearchParams(location.search);
  if (params.get('server')) {
    serverURL = params.get('server');
  }
}

const client = new federated.Client(serverURL, loadAudioTransferLearningModel, {
  verbose: true,
  sendMetrics: true
});

client.on('new-version', () => {
  ui.setVersion(client.numVersions());
});

client.setup().then(async () => {
  let yTrue, yPred, xNpy;

  // Get microphone access
  ui.setStatus('Waiting for microphone&hellip;');
  const stream =
      await navigator.mediaDevices.getUserMedia({audio: true, video: false});

  // On each audio frame, update our rotating buffer with the latest FFT data
  // from our analyser. For fun, also update the spectrum plot
  const numFrames = client.inputShape[0];
  const fftLength = client.inputShape[1];
  const listener = FrequencyListener(stream, numFrames, fftLength);
  listener.onEachFrame(freqData => ui.plotSpectrum(freqData, fftLength));
  listener.listen();

  // Create a recorder to save the raw .wav file
  const recorder = new MediaStreamRecorder(stream);
  recorder.mimeType = 'audio/wav';
  recorder.ondataavailable = wavBlob => {
    // create audio element with recording
    ui.createAudioElement(wavBlob);

    // Setup .wav and .npy files
    const fname = labelNames[yTrue];
    const npyBlob = new Blob([new Uint8Array(xNpy)]);
    const wavFile =
        new File([wavBlob], `${fname}.wav`, {type: 'audio/wav'});
    const npyFile =
        new File([npyBlob], `${fname}.npy`, {type: 'application/octet-stream'});

    // Send data to server
    const req = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('wav', wavFile);
    formData.append('npy', npyFile);
    formData.append('clientId', client.clientId);
    formData.append('modelVersion', client.modelVersion());
    formData.append('timestamp', new Date().getTime().toString());
    formData.append('predLabel', yPred);
    formData.append('trueLabel', yTrue);
    req.open('POST', serverURL + '/data');
    req.send(formData);
  };

  // Setup record buttons
  ui.setReadyStatus(client);
  ui.onRecordButton(labelName => {
    yTrue = labelNames.indexOf(labelName);
    recorder.start(1100);
    setTimeout(finishRecording, 1000);
  })

  // When we're done recording,
  async function finishRecording() {
    // Setup results html
    ui.setupResults();

    // Compute input and label tensors
    const freqData = listener.getFrequencyData();
    const x = getInputTensorFromFrequencyData(freqData, numFrames, fftLength);
    const y = tf.tensor2d([onehot(yTrue)]);
    xNpy = await npy.serialize(x);

    // Compute predictions
    const probs = tf.tidy(() => {
      const p = client.predict(x);
      yPred = tf.argMax(p, 1).dataSync()[0];
      return p.dataSync();
    });

    // Stop separate .wav recording
    recorder.stop();

    // Add plots / result descriptions / magic
    ui.plotSpectrograms(freqData, fftLength);
    ui.plotProbabilities(probs);
    ui.describeResults(yTrue, yPred);
    ui.castSpell(yPred);

    // Train and prepare our next loop!
    ui.setStatus('Fitting Model&hellip;');
    client.federatedUpdate(x, y).catch(console.log).finally(() => {
      tf.dispose([x, y]);
      ui.setReadyStatus(client);
    });
  }
});

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

function onehot(y) {
  const arr = new Array(labelNames.length).fill(0);
  arr[y] = 1;
  return arr;
}
