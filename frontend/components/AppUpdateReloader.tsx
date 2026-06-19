import { useEffect } from 'react';
import * as Updates from 'expo-updates';

import { shouldFetchUpdate } from '../services/appUpdates';

export function AppUpdateReloader() {
  useEffect(() => {
    let isMounted = true;

    async function reloadForAvailableUpdate() {
      if (__DEV__) {
        return;
      }

      try {
        const update = await Updates.checkForUpdateAsync();

        if (!isMounted || !shouldFetchUpdate({ isDevelopment: __DEV__, isAvailable: update.isAvailable })) {
          return;
        }

        await Updates.fetchUpdateAsync();

        if (isMounted) {
          await Updates.reloadAsync();
        }
      } catch (error) {
        console.warn('Failed to check for app updates:', error);
      }
    }

    reloadForAvailableUpdate();

    return () => {
      isMounted = false;
    };
  }, []);

  return null;
}
