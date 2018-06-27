import * as path from 'path';
import {AudioTransferLearningModel} from '../index';
import {setup} from './server';

const dataDir = path.resolve(__dirname + '/../../data');
const model = new AudioTransferLearningModel();
setup(model, dataDir);
