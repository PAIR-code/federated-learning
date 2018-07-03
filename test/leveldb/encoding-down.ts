/* tslint:disable */
declare module 'encoding-down' {
import {AbstractLevelDOWN, AbstractIteratorOptions, AbstractIterator, AbstractOpenOptions, AbstractPutOptions, AbstractGetOptions, AbstractDelOptions, AbstractBatchOptions, ErrorCallback, ErrorValueCallback, AbstractChainedBatch, AbstractBatch, ObjectAny} from 'abstract-leveldown';

import {CodecOptions, CodecEncoder} from 'level-codec';

  export interface EncodingDown<K = any, V = any> extends
      AbstractLevelDOWN<K, V> {
    get(key: K, cb: ErrorValueCallback<V>): void;
    get(key: K, options: EncodingDownGetOptions,
        cb: ErrorValueCallback<V>): void;

    put(key: K, value: V, cb: ErrorCallback): void;
    put(key: K, value: V, options: EncodingDownPutOptions,
        cb: ErrorCallback): void;

    del(key: K, cb: ErrorCallback): void;
    del(key: K, options: EncodingDownDelOptions, cb: ErrorCallback): void;

    batch(): EncodingDownChainedBatch;
    batch(array: AbstractBatch[], cb: ErrorCallback): EncodingDownChainedBatch;
    batch(
        array: AbstractBatch[], options: EncodingDownBatchOptions,
        cb: ErrorCallback): EncodingDownChainedBatch;

    iterator(options?: EncodingDownIteratorOptions): AbstractIterator;
  }

  export interface EncodingDownGetOptions extends AbstractGetOptions,
                                                  CodecOptions {}
  export interface EncodingDownPutOptions extends AbstractPutOptions,
                                                  CodecOptions {}
  export interface EncodingDownDelOptions extends AbstractDelOptions,
                                                  CodecOptions {}
  export interface EncodingDownBatchOptions extends AbstractBatchOptions,
                                                    CodecOptions {}
  export interface EncodingDownIteratorOptions extends AbstractIteratorOptions,
                                                       CodecOptions {}

  export interface EncodingDownChainedBatch<K = any, V = any> extends
      AbstractChainedBatch<K, V>{write(cb: any): any
    write(options: CodecOptions & ObjectAny, cb: any): any
  }

  interface EncodingDOWNConstructor {
      (db: AbstractLevelDOWN, options?: CodecOptions): EncodingDown
          new(db: AbstractLevelDOWN, options?: CodecOptions): EncodingDown
    }

    const EncodingDOWN: EncodingDOWNConstructor;
    export default EncodingDOWN;
}
