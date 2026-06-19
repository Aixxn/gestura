import assert from 'node:assert/strict';
import test from 'node:test';

import { enqueueLiveFrame } from './frameQueuePolicy.ts';

const frame = (id) => ({
  id,
  path: `/tmp/${id}.jpg`,
  timestamp: Number(id.replace('frame_', '')),
});

test('live capture keeps a bounded ordered frame queue', () => {
  const queue = [frame('frame_1'), frame('frame_2')];

  enqueueLiveFrame(queue, frame('frame_3'), {
    maxQueueSize: 3,
    preserveBacklog: false,
  });

  assert.deepEqual(queue.map((item) => item.id), [
    'frame_1',
    'frame_2',
    'frame_3',
  ]);
});

test('live capture drops oldest frames when the bounded queue is full', () => {
  const queue = [frame('frame_1'), frame('frame_2'), frame('frame_3')];

  enqueueLiveFrame(queue, frame('frame_4'), {
    maxQueueSize: 3,
    preserveBacklog: false,
  });

  assert.deepEqual(queue.map((item) => item.id), [
    'frame_2',
    'frame_3',
    'frame_4',
  ]);
});

test('stopping capture preserves queued frames for finalization', () => {
  const queue = [frame('frame_1'), frame('frame_2')];

  enqueueLiveFrame(queue, frame('frame_3'), {
    maxQueueSize: 2,
    preserveBacklog: true,
  });

  assert.deepEqual(queue.map((item) => item.id), [
    'frame_1',
    'frame_2',
    'frame_3',
  ]);
});
