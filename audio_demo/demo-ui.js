const statusDisplay = document.getElementById('status-display');

function logToStatusDisplay(message) {
  const date = new Date();
  statusDisplay.value += `[${date.toISOString()}] ` + message + '\n';
  statusDisplay.scrollTop = statusDisplay.scrollHeight;
}

function plotPredictions(canvas, candidateWords, probabilities) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (probabilities == null) {
    return;
  }

  const barWidth = canvas.width / candidateWords.length * 0.8;
  const barGap = canvas.width / candidateWords.length * 0.2;

  ctx.font = '12px Arial';
  ctx.beginPath();
  for (let i = 0; i < candidateWords.length; ++i) {
    ctx.fillText(
      candidateWords[i], i * (barWidth + barGap), 0.95 * predictionCanvas.height);
  }
  ctx.stroke();

  ctx.beginPath();
  for (let i = 0; i < probabilities.length; ++i) {
    const x = i * (barWidth + barGap);
    ctx.rect(
      x,
      predictionCanvas.height * 0.85 * (1 - probabilities[i]),
      barWidth,
      predictionCanvas.height * 0.85 * probabilities[i]);
  }
  ctx.stroke();
}

function setUpThresholdSlider(runOptions) {
  const thresholdSlider = document.getElementById('magnitude-threshold');
  thresholdSlider.setAttribute('min', runOptions.magnitudeThresholdMin);
  thresholdSlider.setAttribute('max', runOptions.magnitudeThresholdMax);

  const magnitudeThresholdSpan =
      document.getElementById('magnitude-threshold-span');
  thresholdSlider.value = runOptions.magnitudeThreshold;
  magnitudeThresholdSpan.textContent = runOptions.magnitudeThreshold;
  thresholdSlider.addEventListener('click', () => {
    runOptions.magnitudeThreshold = thresholdSlider.value;
    magnitudeThresholdSpan.textContent = runOptions.magnitudeThreshold;
  });

  const magnitudeThresholdInc =
      document.getElementById('magnitude-threshold-increase');
  const magnitudeThresholdDec =
      document.getElementById('magnitude-threshold-decrease');
  magnitudeThresholdInc.addEventListener('click', () => {
    if (runOptions.magnitudeThreshold + 1 > runOptions.magnitudeThresholdMax) {
      return;
    }
    runOptions.magnitudeThreshold++;
    thresholdSlider.value = runOptions.magnitudeThreshold;
    magnitudeThresholdSpan.textContent = runOptions.magnitudeThreshold;
  });
  magnitudeThresholdDec.addEventListener('click', () => {
    if (runOptions.magnitudeThreshold - 1 < runOptions.magnitudeThresholdMin) {
      return;
    }
    runOptions.magnitudeThreshold--;
    thresholdSlider.value = runOptions.magnitudeThreshold;
    magnitudeThresholdSpan.textContent = runOptions.magnitudeThreshold;
  });
}

function plotSpectrum(canvas, freqData, runOptions) {
  let instanceMax = -Infinity;
  for (const val of freqData) {
    if (val > instanceMax) {
      instanceMax = val;
    }
  }
  const yOffset = 0.1 * canvas.height;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle =
    instanceMax > runOptions.magnitudeThreshold ? '#00AA00' : '#AAAAAA';
  ctx.lineWidth = 1;
  ctx.beginPath();

  ctx.moveTo(0, -freqData[0] + yOffset);
  for (let i = 1; i < runOptions.modelFFTLength; ++i) {
    ctx.lineTo(i, -freqData[i] + yOffset);
  }
  ctx.stroke();

  // Draw the threshold.
  ctx.beginPath();
  ctx.moveTo(0, -runOptions.magnitudeThreshold + yOffset);
  ctx.lineTo(
    runOptions.modelFFTLength - 1, -runOptions.magnitudeThreshold + yOffset);
  ctx.stroke();
}

