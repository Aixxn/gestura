export interface UpdateAvailability {
  isDevelopment: boolean;
  isAvailable: boolean;
}

export function shouldFetchUpdate({ isDevelopment, isAvailable }: UpdateAvailability) {
  return !isDevelopment && isAvailable;
}
