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

import {FederatedModel, FederatedTfModel} from './common';

/**
 * Interface that FederatedClientModels must support. Essentially a wrapper
 * around FederatedModel, which is defined in src/common/index.ts.
 */
export interface FederatedClientModel extends FederatedModel {
  isFederatedClientModel: boolean;

  setup(): Promise<void>;
}

/**
 * Specific version of FederatedClientModel that wraps a `tf.Model`, async
 * function returning a `tf.Model`, or a string that can be passed to
 * `tf.loadModel`.
 */
export class FederatedClientTfModel extends FederatedTfModel implements
    FederatedClientModel {
  isFederatedClientModel = true;

  async setup() {
    await this.fetchInitial();
  }
}

/**
 * Type guard for FederatedClientModel.
 *
 * @param model any object
 */
// tslint:disable-next-line:no-any
export function isFederatedClientModel(model: any):
    model is FederatedClientModel {
  return model && model.isFederatedClientModel;
}
