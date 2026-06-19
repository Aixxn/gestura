export interface FrameQueueItem {
  id: string;
  path: string;
  timestamp: number;
}

interface EnqueueOptions {
  maxQueueSize: number;
  preserveBacklog: boolean;
}

export const enqueueLiveFrame = <T extends FrameQueueItem>(
  queue: T[],
  frame: T,
  options: EnqueueOptions
) => {
  queue.push(frame);

  if (options.preserveBacklog) {
    return;
  }

  const overflowCount = queue.length - options.maxQueueSize;
  if (overflowCount > 0) {
    queue.splice(0, overflowCount);
  }
};
