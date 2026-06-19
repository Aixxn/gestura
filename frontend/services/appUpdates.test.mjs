import assert from 'node:assert/strict';
import test from 'node:test';

import { shouldFetchUpdate } from './appUpdates.ts';

test('fetches updates only outside development when an update is available', () => {
  assert.equal(shouldFetchUpdate({ isDevelopment: false, isAvailable: true }), true);
  assert.equal(shouldFetchUpdate({ isDevelopment: false, isAvailable: false }), false);
  assert.equal(shouldFetchUpdate({ isDevelopment: true, isAvailable: true }), false);
});
