import { useSyncExternalStore } from 'react';

import {
  getTranslationHistory,
  subscribeToTranslationHistory,
} from '../services/translationHistory';

export const useTranslationHistory = () => (
  useSyncExternalStore(
    subscribeToTranslationHistory,
    getTranslationHistory,
    getTranslationHistory
  )
);
