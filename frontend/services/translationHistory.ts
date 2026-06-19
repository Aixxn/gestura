export interface TranslationHistoryEntry {
  id: string;
  createdAt: string;
  aslGloss: string;
  english: string;
  videoPath: string;
}

export interface NewTranslationHistoryEntry {
  aslGloss: string;
  english: string;
  videoPath: string;
}

type Listener = () => void;

const listeners = new Set<Listener>();
let entries: TranslationHistoryEntry[] = [];

const createId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

const notify = () => {
  listeners.forEach((listener) => listener());
};

export const addTranslationHistoryEntry = (
  entry: NewTranslationHistoryEntry
): TranslationHistoryEntry => {
  const storedEntry: TranslationHistoryEntry = {
    id: createId(),
    createdAt: new Date().toISOString(),
    aslGloss: entry.aslGloss.trim(),
    english: entry.english.trim(),
    videoPath: entry.videoPath,
  };

  entries = [storedEntry, ...entries];
  notify();
  return storedEntry;
};

export const getTranslationHistory = () => entries;

export const clearTranslationHistory = () => {
  entries = [];
  notify();
};

export const subscribeToTranslationHistory = (listener: Listener) => {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
};
