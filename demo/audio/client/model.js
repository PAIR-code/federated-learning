import * as tf from '@tensorflow/tfjs';

export const labelNames = [
  'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
  'zero', 'left', 'right', 'go', 'stop'
];

const audioTransferLearningModelURL =
    'https://storage.googleapis.com/tfjs-speech-command-model-14w/model.json';

export async function loadAudioTransferLearningModel() {
  const model = await tf.loadModel(audioTransferLearningModelURL);

  for (let i = 0; i < 9; ++i) {
    model.layers[i].trainable = false;  // freeze conv layers
  }

  model.compile({'optimizer': 'sgd', loss: 'categoricalCrossentropy'});

  return model;
}
