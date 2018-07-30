import * as tf from '@tensorflow/tfjs';
import {labelNames, loadAudioTransferLearningModel} from './model';
import * as fs from 'fs';
import * as path from 'path';
import * as npy from './npy';

function parseNpyFile(name): tf.Tensor {
  const buff = fs.readFileSync(path.resolve(__dirname + '/' + name));
  const arrayBuff =
      buff.buffer.slice(buff.byteOffset, buff.byteOffset + buff.byteLength);
  return npy.parse(arrayBuff);
}

const validInputs = parseNpyFile('hp-validation-inputs.npy');
const validLabels = tf.tidy(
    () =>
        tf.oneHot(parseNpyFile('hp-validation-labels.npy') as tf.Tensor1D, 4));

const trainInputs = parseNpyFile('val-inputs.npy');
const trainLabels = tf.tidy(
    () =>
        tf.oneHot(parseNpyFile('val-labels.npy') as tf.Tensor1D, 4));

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
