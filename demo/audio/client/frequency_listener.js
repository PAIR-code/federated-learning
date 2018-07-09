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

export function FrequencyListener(stream, numFrames, modelFFTLength) {
  const frameSize = 1024;
  const rotatingBufferSizeMultiplier = 2;
  let listener = {};
  let frameCount = 0;
  const callbacks = [];

  const AudioContext = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = frameSize * 2;
  analyser.smoothingTimeConstant = 0.0;
  source.connect(analyser);

  const freqData = new Float32Array(analyser.frequencyBinCount);
  const rotatingBufferNumFrames =
      numFrames * rotatingBufferSizeMultiplier;
  const rotatingBufferSize =
      modelFFTLength * rotatingBufferNumFrames;
  const rotatingBuffer = new Float32Array(rotatingBufferSize);

  const handleAudioFrame = () => {
    analyser.getFloatFrequencyData(freqData);
    if (freqData[0] === -Infinity && freqData[1] === -Infinity) {
      return;
    }
    const freqDataSlice = freqData.slice(0, modelFFTLength);
    for (let i = 0; i < callbacks.length; i++) {
      callbacks[i](freqDataSlice);
    }
    const bufferPos = frameCount % rotatingBufferNumFrames;
    rotatingBuffer.set(freqDataSlice, bufferPos * modelFFTLength);
    frameCount++;
  }

  listener.listen = function() {
    setInterval(
        handleAudioFrame,
        analyser.frequencyBinCount / audioContext.sampleRate * 1000);
  }

  listener.onEachFrame = function(callback) {
    callbacks.push(callback);
  }

  listener.getFrequencyData = function() {
    const size = numFrames * modelFFTLength;
    const data = new Float32Array(size);

    let fc = frameCount - numFrames;
    while (fc < 0) {
      fc += rotatingBufferNumFrames;
    }

    const indexBegin = (fc % rotatingBufferNumFrames) * modelFFTLength;
    const indexEnd = indexBegin + size;
    for (let i = indexBegin; i < indexEnd; ++i) {
      data[i - indexBegin] = rotatingBuffer[i % rotatingBufferSize];
    }
    return data;
  }

  return listener;
}
