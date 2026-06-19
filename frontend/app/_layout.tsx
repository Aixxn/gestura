import { DarkTheme, ThemeProvider } from '@react-navigation/native';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { AppUpdateReloader } from '../components/AppUpdateReloader';

export const unstable_settings = {
  // Ensure index is the landing page
  initialRouteName: '(tabs)/index',
};

export default function RootLayout() {

  return (
    <ThemeProvider value={DarkTheme}>
      <AppUpdateReloader />
      <Stack
        screenOptions={{
          headerShown: false,
        }}
      >
        {/* Define screens in the stack */}
      </Stack>
      <StatusBar style="auto" />
    </ThemeProvider>
  );
}
