import * as tf from '@tensorflow/tfjs';

import {FederatedModel, FederatedTfModel} from './common';

// tslint:disable-next-line:max-line-length
import {AsyncTfModel, clientHyperparams, ClientHyperparams, deserializeVars, DownloadMsg, Events, FederatedCompileConfig, SerializedVariable, serializeVars, serverHyperparams, ServerHyperparams, stackSerialized, UploadCallback, UploadMsg, VersionCallback} from './common';
// tslint:disable-next-line:max-line-length

import {MockitIOServer} from './mockit_io';
/**
 * FederatedServerModel describes the interface that models passed to `Server`
 * must implement.
 *
 * See the FederatedModel documentation in src/common/index.ts for more details.
 */
export interface FederatedServerModel extends FederatedModel {
  isFederatedServerModel: boolean;
  version: string;

  /**
   * Initialize the model
   */
  setup(): Promise<void>;

  /**
   * Save the current model and update `version`.
   */
  save(): Promise<void>;
}

/**
 * Type guard for federated server models.
 *
 * @param model any object
 */
// tslint:disable-next-line:no-any
export function isFederatedServerModel(model: any):
    model is FederatedServerModel {
  return model && model.isFederatedServerModel;
}

export class FederatedServerInMemoryModel extends FederatedTfModel implements
    FederatedServerModel {
  isFederatedServerModel = true;
  version: string;

  async setup() {
    await this.fetchInitial();
    await this.save();
  }

  async save() {
    this.version = new Date().getTime().toString();
  }
}

export type FederatedServerConfig = {
  clientHyperparams?: ClientHyperparams,
  serverHyperparams?: ServerHyperparams,
  updatesPerVersion?: number,
  modelDir?: string,
  modelCompileConfig?: FederatedCompileConfig,
  verbose?: boolean
};

/**
 * Federated Learning Server library.
 *
 * Example usage with a tf.Model:
 * ```js
 * const model = await tf.loadModel('file:///a/model.json');
 * const webServer = http.createServer();
 * const fedServer = new Server(webServer, model);
 * fedServer.setup().then(() => {
 *  webServer.listen(80);
 * });
 * ```
 *
 * The server aggregates model weight updates from clients and publishes new
 * versions of the model periodically to all clients.
 */
export class MockServer {
  model: FederatedServerModel;
  clientHyperparams: ClientHyperparams;
  serverHyperparams: ServerHyperparams;
  downloadMsg: DownloadMsg;
  server: MockitIOServer;
  numClients = 0;
  numUpdates = 0;
  updates: SerializedVariable[][] = [];
  updating = false;
  versionCallbacks: VersionCallback[];
  uploadCallbacks: UploadCallback[];
  verbose: boolean;

  constructor(
      webServer: MockitIOServer, model: AsyncTfModel|FederatedServerModel,
      config: FederatedServerConfig) {
    // Setup server
    this.server = webServer;

    // Setup model
    if (isFederatedServerModel(model)) {
      this.model = model;
    } else {
      const compileConfig = config.modelCompileConfig || {};
      this.model = new FederatedServerInMemoryModel(model, compileConfig);
    }
    this.verbose = (!!config.verbose) || false;
    this.clientHyperparams = clientHyperparams(config.clientHyperparams || {});
    this.serverHyperparams = serverHyperparams(config.serverHyperparams || {});
    this.downloadMsg = null;
    this.uploadCallbacks = [];
    this.versionCallbacks = [(v1, v2) => {
      this.log(`updated model: ${v1} -> ${v2}`);
    }];
  }

  /**
   * Set up the federated learning server.
   *
   * This mainly delegates to `FederatedServerModel.setup` but also performs
   * any user-defined callbacks and initializes the websocket server.
   */
  async setup() {
    await this.time('setting up model', async () => {
      await this.model.setup();
    });

    this.downloadMsg = await this.computeDownloadMsg();
    await this.performCallbacks();

    this.server.on('connection', (socket) => {
      this.numClients++;
      this.log(`connection: ${this.numClients} clients`);

      socket.on('disconnect', () => {
        this.numClients--;
        this.log(`disconnection: ${this.numClients} clients`);
      });

      socket.emit(Events.Download, this.downloadMsg);

      socket.on(Events.Upload, async (msg: UploadMsg, ack) => {
        if (ack) {
          ack(true);
        }
        if (msg.model.version === this.model.version && !this.updating) {
          this.log(`new update from ${msg.clientId}`);
          this.updates.push(msg.model.vars);
          this.numUpdates++;
          await this.time('upload callbacks', async () => {
            this.uploadCallbacks.forEach(c => c(msg));
          });
          if (this.shouldUpdate()) {
            await this.updateModel();
            this.server.sockets.emit(Events.Download, this.downloadMsg);
          }
        }
      });
    });
  }

  private shouldUpdate(): boolean {
    const numUpdates = this.numUpdates;
    return (numUpdates >= this.serverHyperparams.minUpdatesPerVersion);
  }

  /**
   * Register a new callback to be invoked whenever the server updates the model
   * version.
   *
   * @param callback function to be called (w/ old and new version IDs)
   */
  onNewVersion(callback: VersionCallback) {
    this.versionCallbacks.push(callback);
  }

  /**
   * Register a new callback to be invoked whenever the client uploads a new set
   * of weights.
   *
   * @param callback function to be called (w/ client's upload msg)
   */
  onUpload(callback: UploadCallback) {
    this.uploadCallbacks.push(callback);
  }

  private async computeDownloadMsg(): Promise<DownloadMsg> {
    return {
      model: {
        vars: await serializeVars(this.model.getVars()),
        version: this.model.version,
      },
      hyperparams: this.clientHyperparams
    };
  }

  // TODO: optionally clip updates by global norm
  // TODO: implement median and trimmed mean aggregations
  // TODO: optionally skip updates if validation loss increases
  // TOOD: consider only updating once we achieve a certain number of _clients_
  private async updateModel() {
    this.updating = true;
    const oldVersion = this.model.version;
    const aggregation = this.serverHyperparams.aggregation;

    await this.time('computing new weights', async () => {
      const newWeights = tf.tidy(() => {
        const stacked = stackSerialized(this.updates);
        const updates = deserializeVars(stacked);
        if (aggregation === 'mean') {
          return updates.map(update => update.mean(0));
        } else {
          throw new Error(`unsupported aggregation ${aggregation}`);
        }
      });
      this.model.setVars(newWeights);
    });

    this.model.save();
    this.downloadMsg = await this.computeDownloadMsg();
    this.updates = [];
    this.numUpdates = 0;
    this.updating = false;
    this.performCallbacks(oldVersion);
  }

  // tslint:disable-next-line:no-any
  private log(...args: any[]) {
    if (this.verbose) {
      console.log('Federated Server:', ...args);
    }
  }

  private async time(msg: string, action: () => Promise<void>) {
    const t1 = new Date().getTime();
    await action();
    const t2 = new Date().getTime();
    this.log(`${msg} took ${t2 - t1}ms`);
  }

  private async performCallbacks(oldVersion?: string) {
    await this.time('performing callbacks', async () => {
      this.versionCallbacks.forEach(c => c(oldVersion, this.model.version));
    });
  }
}
