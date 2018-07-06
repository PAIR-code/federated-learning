export function FrequencyListener(stream, runOptions) {
  let listener = {};
  let frameCount = 0;
  const callbacks = [];

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

  const handleAudioFrame = () => {
    analyser.getFloatFrequencyData(freqData);
    if (freqData[0] === -Infinity && freqData[1] === -Infinity) {
      return;
    }
    const freqDataSlice = freqData.slice(0, runOptions.modelFFTLength);
    for (let i = 0; i < callbacks.length; i++) {
      callbacks[i](freqDataSlice);
    }
    const bufferPos = frameCount % rotatingBufferNumFrames;
    rotatingBuffer.set(freqDataSlice, bufferPos * runOptions.modelFFTLength);
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
    return getFrequencyDataFromRotatingBuffer(
        rotatingBuffer, frameCount - runOptions.numFrames, runOptions);
  }

  return listener;
}

function getFrequencyDataFromRotatingBuffer(
    rotatingBuffer, frameCount, runOptions) {
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
    freqData[i - indexBegin] = rotatingBuffer[i % rotatingBufferSize];
  }
  return freqData;
}
