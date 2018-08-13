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

import {labelNames} from './model';
import * as spectralPlots from './spectral_plots';

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

let resultRow;

export function onRecordButton(callback) {
  recordButtons.forEach(button => {
    button.addEventListener('click', () => {
      button.classList.add('active');
      lastPrediction.innerHTML = '';
      recordButtons.forEach(b => b.setAttribute('disabled', 'disabled'));
      modelDiv.innerHTML = waitingTemplate;
      setStatus('Listening&hellip;');
      callback(button.getAttribute('value'));
    })
  })
}

export function setStatus(statusHTML) {
  statusBar.innerHTML = statusHTML;
}

export function setVersion(version) {
  modelVersion.innerText = `version #${version}`;
}

export function setReadyStatus(client) {
  recordButtons.forEach(b => {
    b.removeAttribute('disabled');
    b.classList.remove('active');
  });

  let html = 'Ready to listen!';
  const nx = client.numExamplesRemaining();
  html += ` Need <span class='status-number'>${nx}</span> more before training.`
  const trained = client.numUpdates();
  if (trained) {
    html += ` You've trained <span class='status-number'>${trained}</span> 🧙`;
  }
  setStatus(html);
}

export function setupResults() {
  modelDiv.innerHTML = modelTemplate;
  resultRow = document.createElement('tr');
  for (let i = 0; i < 4; i++)
    resultRow.appendChild(document.createElement('td'));
  labeledExamples.prepend(resultRow);
}

export function describeResults(yTrue, yPred) {
  resultRow.children[2].innerText = labelNames[yTrue];
  resultRow.children[3].innerText = labelNames[yPred];
  if (yTrue != yPred) {
    resultRow.children[3].className = 'incorrect-prediction';
    lastPrediction.innerHTML = `You tried to cast ` +
      `<span class='last-prediction'>${labelNames[yTrue]}</span>, ` +
      `but your wand performed ` +
      `<span class='last-prediction'>${labelNames[yPred]}</span>!`;
  } else {
    lastPrediction.innerHTML = `You successfully cast ` +
      `<span class='last-prediction'>${labelNames[yPred]}</span>!`;
  }
}

export function plotSpectrum(freqData, fftLength) {
  spectralPlots.plotSpectrum(spectrumCanvas, freqData, fftLength);
}

export function plotSpectrograms(freqData, fftLength) {
  const mainSpectrogram = document.getElementById('spectrogram-canvas');
  const miniSpectrogram = document.createElement('canvas');
  miniSpectrogram.setAttribute('width', '81');
  miniSpectrogram.setAttribute('height', '54');
  spectralPlots.plotSpectrogram(mainSpectrogram, freqData, fftLength);
  spectralPlots.plotSpectrogram(miniSpectrogram, freqData, fftLength);
  resultRow.children[1].appendChild(miniSpectrogram);
}

export function plotProbabilities(probs) {
  Plotly.newPlot('probs', [{x: labelNames, y: probs, type: 'bar'}], {
    autosize: false,
    width: Math.min(270*1.25, modelDiv.clientWidth),
    height: 180,
    margin: {l: 30, r: 5, b: 30, t: 5, pad: 0},
  });
}

export function createAudioElement(wavBlob) {
  const url = URL.createObjectURL(wavBlob);
  const audioControls = document.createElement('audio');
  const audioSource = document.createElement('source');
  audioControls.setAttribute('controls', 'controls');
  audioSource.src = url;
  audioSource.type = 'audio/wav';
  audioControls.appendChild(audioSource);
  resultRow.children[0].appendChild(audioControls);
}

export function castSpell(spell) {
  if (spell === 'nox') {
    htmlEl.classList.add('nox');
  } else if (spell === 'lumos') {
    htmlEl.classList.remove('nox');
  } else if (spell === 'accio') {
    htmlEl.classList.add('accio');
  } else if (spell === 'expelliarmus') {
    htmlEl.classList.remove('accio');
  } else {
    console.log(`Unknown spell ${spell}`);
  }
}
