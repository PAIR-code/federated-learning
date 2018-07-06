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

export function plotSpectrum(canvas, freqData, modelFFTLength) {
  let instanceMax = -Infinity;
  for (const val of freqData) {
    if (val > instanceMax) {
      instanceMax = val;
    }
  }
  const yscale = 0.75;
  const xscale = canvas.width / modelFFTLength;
  const yOffset = -0.1 * canvas.height;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#AAAAAA';
  ctx.lineWidth = 1;
  ctx.beginPath();

  ctx.moveTo(0, -freqData[0]*yscale + yOffset);
  for (let i = 1; i < modelFFTLength; ++i) {
    ctx.lineTo(i*xscale, -freqData[i]*yscale + yOffset);
  }
  ctx.stroke();
}

export function plotSpectrogram(canvas, frequencyData, fftSize) {
  // Get the maximum and minimum.
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < frequencyData.length; ++i) {
    const x = frequencyData[i];
    if (x !== -Infinity) {
      if (x < min) {
        min = x;
      }
      if (x > max) {
        max = x;
      }
    }
  }
  if (min >= max) {
    return;
  }

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const numTimeSteps = frequencyData.length / fftSize;
  const pixelWidth = canvas.width / numTimeSteps;
  const pixelHeight = canvas.height / fftSize;
  for (let i = 0; i < numTimeSteps; ++i) {
    const x = pixelWidth * i;
    const spectrum = frequencyData.subarray(i * fftSize, (i + 1) * fftSize);
    if (spectrum[0] === -Infinity) {
      break;
    }
    for (let j = 0; j < fftSize; ++j) {
      const y = canvas.height - (j + 1) * pixelHeight;

      let colorValue = (spectrum[j] - min) / (max - min);
      colorValue = Math.pow(colorValue, 3);
      colorValue = Math.round(255 * colorValue);
      const fillStyle = `rgb(${colorValue},${255 - colorValue},${255 - colorValue})`;
      ctx.fillStyle = fillStyle;
      ctx.fillRect(x, y, pixelWidth, pixelHeight);
    }
  }
}
