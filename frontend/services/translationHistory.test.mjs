import assert from 'node:assert/strict';
import test from 'node:test';

import {
  addTranslationHistoryEntry,
  clearTranslationHistory,
  getTranslationHistory,
} from './translationHistory.ts';

test('stores translation history entries newest first', () => {
  clearTranslationHistory();

  const first = addTranslationHistoryEntry({
    aslGloss: 'ME HUNGRY',
    english: 'I am hungry.',
    videoPath: '/tmp/first.mp4',
  });
  const second = addTranslationHistoryEntry({
    aslGloss: 'HELLO WORLD',
    english: 'Hello, world.',
    videoPath: '/tmp/second.mp4',
  });

  assert.deepEqual(
    getTranslationHistory().map((entry) => entry.id),
    [second.id, first.id]
  );
});

test('clearTranslationHistory removes all entries', () => {
  clearTranslationHistory();
  addTranslationHistoryEntry({
    aslGloss: 'THANK YOU',
    english: 'Thank you.',
    videoPath: '/tmp/thanks.mp4',
  });

  clearTranslationHistory();

  assert.deepEqual(getTranslationHistory(), []);
});

test('getTranslationHistory returns the same snapshot while history is unchanged', () => {
  clearTranslationHistory();
  addTranslationHistoryEntry({
    aslGloss: 'HELLO',
    english: 'Hello.',
    videoPath: '/tmp/hello.mp4',
  });

  const firstSnapshot = getTranslationHistory();
  const secondSnapshot = getTranslationHistory();

  assert.equal(firstSnapshot, secondSnapshot);
});
