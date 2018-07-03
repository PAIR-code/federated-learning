/* tslint:disable */
declare module 'leveldown' {
import {AbstractLevelDOWN, AbstractIteratorOptions, AbstractIterator, AbstractOpenOptions, AbstractPutOptions, AbstractGetOptions, AbstractDelOptions, AbstractBatchOptions, ErrorCallback, ErrorValueCallback, AbstractChainedBatch, AbstractBatch} from 'abstract-leveldown';

  export type Bytes = string|Buffer;
  export type ErrorSizeCallback = (err: Error|undefined, size: number) => void;

  export interface LevelDown extends AbstractLevelDOWN<Bytes, Bytes> {
    open(cb: ErrorCallback): void;
    open(options: LevelDownOpenOptions, cb: ErrorCallback): void;

    get(key: Bytes, cb: ErrorValueCallback<Bytes>): void;
    get(key: Bytes, options: LevelDownGetOptions,
        cb: ErrorValueCallback<Bytes>): void;

    put(key: Bytes, value: Bytes, cb: ErrorCallback): void;
    put(key: Bytes, value: Bytes, options: LevelDownPutOptions,
        cb: ErrorCallback): void;

    del(key: Bytes, cb: ErrorCallback): void;
    del(key: Bytes, options: LevelDownDelOptions, cb: ErrorCallback): void;

    batch(): AbstractChainedBatch<Bytes, Bytes>;
    batch(array: AbstractBatch[], cb: ErrorCallback):
        AbstractChainedBatch<Bytes, Bytes>;
    batch(
        array: AbstractBatch[], options: LevelDownBatchOptions,
        cb: ErrorCallback): AbstractChainedBatch<Bytes, Bytes>;

    approximateSize(start: Bytes, end: Bytes, cb: ErrorSizeCallback): void;
    compactRange(start: Bytes, end: Bytes, cb: ErrorCallback): void;
    getProperty(property: string): string;
    destroy(location: string, cb: ErrorCallback): void;
    repair(location: string, cb: ErrorCallback): void;
    iterator(options?: LevelDownIteratorOptions): LevelDownIterator
  }

  interface LevelDownConstructor {
    new(location: string): LevelDown;
    (location: string): LevelDown;
  }

  export interface LevelDownOpenOptions extends AbstractOpenOptions {
    compression?: boolean;
    cacheSize?: number;
    writeBufferSize?: number;
    blockSize?: number;
    maxOpenFiles?: number;
    blockRestartInterval?: number;
    maxFileSize?: number;
  }

  export interface LevelDownGetOptions extends AbstractGetOptions {
    fillCache?: boolean;
  }

  export interface LevelDownPutOptions extends AbstractPutOptions {
    sync?: boolean
  }

  export interface LevelDownDelOptions extends AbstractDelOptions {
    sync?: boolean;
  }

  export interface LevelDownBatchOptions extends AbstractBatchOptions {
    sync?: boolean;
  }

  export interface LevelDownIteratorOptions extends
      AbstractIteratorOptions<Bytes> {
    fillCache?: boolean;
  }

  export interface LevelDownIterator extends AbstractIterator<Bytes, Bytes> {
    seek(key: Bytes): void;
    binding: any;
    cache: any;
    finished: any;
    fastFuture: any;
  }

  const LevelDOWN: LevelDownConstructor;
  export default LevelDOWN;
}
