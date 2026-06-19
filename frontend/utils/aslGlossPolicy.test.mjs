import assert from 'node:assert/strict';
import test from 'node:test';

import { appendDetectedWord } from './aslGlossPolicy.ts';

test('appends word_detected responses to raw ASL gloss', () => {
  const gloss = appendDetectedWord('HELLO', {
    status: 'word_detected',
    word: 'WORLD',
  });

  assert.equal(gloss, 'HELLO WORLD');
});

test('ignores duplicate consecutive detected words', () => {
  const gloss = appendDetectedWord('HELLO', {
    status: 'word_detected',
    word: 'HELLO',
  });

  assert.equal(gloss, 'HELLO');
});

test('ignores non-word frame responses', () => {
  const gloss = appendDetectedWord('HELLO', {
    status: 'processing',
  });

  assert.equal(gloss, 'HELLO');
});
