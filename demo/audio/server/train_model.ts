import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import * as npy from './npy';

import fetch from 'node-fetch';
// tslint:disable-next-line:no-any
(global as any).fetch = fetch;

async function loadAudioTransferLearningModel(url: string) {
  // NOTE: have to temporarily pretend that this is a browser
  tf.ENV.set('IS_BROWSER', true);
  const model = await tf.loadModel(url);
  tf.ENV.set('IS_BROWSER', false);

  let final;

  if (url.indexOf('http') >= 0) {
    for (let i = 0; i < model.layers.length; ++i) {
      model.layers[i].trainable = false;  // freeze conv layers
    }
    const cutoffTensor = model.layers[10].output;
    const k = 4;
    const newDenseLayer = tf.layers.dense({units: k, activation: 'softmax'});
    const newOutputTensor = newDenseLayer.apply(cutoffTensor);
    const transferModel = tf.model(
        {inputs: model.inputs, outputs: newOutputTensor as tf.SymbolicTensor});
    transferModel.compile({
      loss: 'categoricalCrossentropy',
      optimizer: 'sgd',
      metrics: ['accuracy']
    });
    final = transferModel;
  } else {
    model.compile({
      'optimizer': 'sgd',
      loss: 'categoricalCrossentropy',
      'metrics': ['accuracy']
    });
    final = model;
  }
  return final;
}

function parseNpyFile(name): tf.Tensor {
  const buff = fs.readFileSync(path.resolve(__dirname + '/' + name));
  const arrayBuff =
      buff.buffer.slice(buff.byteOffset, buff.byteOffset + buff.byteLength);
  return npy.parse(arrayBuff);
}

const validInputs = parseNpyFile('hp-validation-inputs.npy');
const validLabels = tf.tidy( () => tf.oneHot(parseNpyFile('hp-validation-labels.npy') as tf.Tensor1D, 4));
const trainInputs = validInputs;
const trainLabels = validLabels;

        //const trainInputs = parseNpyFile('val-inputs.npy');
        //const trainLabels = tf.tidy( () => tf.oneHot(parseNpyFile('val-labels.npy') as tf.Tensor1D, 4));

loadAudioTransferLearningModel('https://storage.googleapis.com/tfjs-speech-command-model-14w/model.json').then(model => {
  const optimizer = tf.train.sgd(0.001);
  model.compile({
    'optimizer': optimizer,
    loss: 'categoricalCrossentropy',
    'metrics': ['accuracy']
  });
  model.evaluate(trainInputs, trainLabels)[1].print();
  model.evaluate(validInputs, validLabels)[1].print();
  model.fit(trainInputs, trainLabels, {epochs: 10, batchSize: 4}).then(() => {
    model.evaluate(trainInputs, trainLabels)[1].print();
    model.evaluate(validInputs, validLabels)[1].print();
    model.save('file://manual2');
  });
});
