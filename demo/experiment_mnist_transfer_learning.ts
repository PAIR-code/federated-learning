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

import fetch from 'node-fetch';

(global as any).fetch = fetch;

import * as tf from '@tensorflow/tfjs';

import {VariableSynchroniser} from '../src/client/comm';
import {MnistTransferLearningModel} from './mnist_transfer_learning_model';
import {loadMnist} from './mnist_data';

function zip<T, U>(x: T[], y: U[]) {
  return x.map((_, i) => [x[i], y[i]] as [T, U]);
}

async function main(splitStart: number, splitEnd: number) {
  const localExamples = splitEnd - splitStart;
  const fedModel = new MnistTransferLearningModel();
  const {vars, loss} = await fedModel.setup();

  const sync = new VariableSynchroniser(vars);
  const fitConfig = await sync.initialise('http://localhost:3000');
  console.log('connected to server');

  const batchSize = fitConfig.batchSize;

  const {imgs: allImgs, labels: allLabels} = loadMnist();

  const split = <T extends tf.Tensor>(t: T) =>
      tf.slice(t, [splitStart], [splitEnd - splitStart]);

  const [imgs, labels] = [split(allImgs), split(allLabels)];

  if ((localExamples % batchSize) !== 0) {
    throw new Error(
        'local batchsize must exactly divide number of local examples');
  }

  const imgsBatches = tf.split(imgs, localExamples / batchSize);
  const labelsBatches = tf.split(labels, localExamples / batchSize);

  allImgs.dispose();
  allLabels.dispose();
  imgs.dispose();
  labels.dispose();

  const optimizer = tf.train.sgd(0.01);

  for (const [img, label] of zip(imgsBatches, labelsBatches)) {
    optimizer.minimize(() => loss(img, tf.oneHot(label, 10)));
    sync.numExamples += batchSize;
    await sync.uploadVars();
    console.log('synced');
  }
}

(async () => {
  try {
    main(0, 16);
  } catch (exn) {
    console.error(exn);
  }
})();
