import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import {
  Image,
  ImageBackground,
  SafeAreaView,
  StatusBar,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';

import { loginUser } from '../../services/auth';

const Login = () => {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const handleLogin = async () => {
    if (isLoading) return;
    setErrorMessage('');
    setIsLoading(true);
    try {
      await loginUser({ email, password });
      router.push('/(tabs)/camera');
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Login failed. Try again.';
      setErrorMessage(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <View style={styles.page}>
      <StatusBar barStyle="light-content" />
      <ImageBackground
        source={require('../../images/login-night.png')}
        style={styles.background}
        resizeMode="cover"
      >
        <SafeAreaView style={styles.safeArea}>
          <View style={styles.brandRow}>
            <Text style={styles.brandText}>Gestura</Text>
          </View>

          <View style={styles.formCardWrap}>
            <View style={styles.formCard}>
              <Text style={styles.title}>Login</Text>
              <View style={styles.signupRow}>
              <Text style={styles.signupHint}>Don't have an account?</Text>
              <TouchableOpacity onPress={() => router.push('/(tabs)/registration')}>
                <Text style={styles.signupLink}>Sign Up</Text>
              </TouchableOpacity>
            </View>

            {errorMessage ? (
              <Text style={styles.errorText}>{errorMessage}</Text>
            ) : null}

              <View style={styles.fieldBlock}>
                <Text style={styles.label}>Email</Text>
                <View style={styles.inputWrap}>
                  <TextInput
                    value={email}
                    onChangeText={setEmail}
                    placeholder="Loisbecket@gmail.com"
                    placeholderTextColor="#7f8aa4"
                    autoCapitalize="none"
                    keyboardType="email-address"
                    style={styles.input}
                  />
                </View>
              </View>

              <View style={styles.fieldBlock}>
                <Text style={styles.label}>Password</Text>
                <View style={styles.inputWrap}>
                  <TextInput
                    value={password}
                    onChangeText={setPassword}
                    placeholder="********"
                    placeholderTextColor="#7f8aa4"
                    secureTextEntry={true}
                    style={styles.input}
                  />
                  <Image
                    source={require('../../images/eye-off.png')}
                    style={styles.iconRight}
                    resizeMode="contain"
                  />
                </View>
              </View>

              <View style={styles.optionsRow}>
                <TouchableOpacity
                  style={styles.rememberRow}
                  onPress={() => setRememberMe(!rememberMe)}
                >
                  <View
                    style={[
                      styles.checkbox,
                      rememberMe ? styles.checkboxChecked : null,
                    ]}
                  >
                    {rememberMe ? <View style={styles.checkboxInner} /> : null}
                  </View>
                  <Text style={styles.rememberText}>Remember me</Text>
                </TouchableOpacity>
                <TouchableOpacity>
                  <Text style={styles.forgotText}>Forgot Password ?</Text>
                </TouchableOpacity>
              </View>

            <TouchableOpacity style={styles.primaryButton} onPress={handleLogin}>
              <Text style={styles.primaryButtonText}>
                {isLoading ? 'Logging in...' : 'Log In'}
              </Text>
            </TouchableOpacity>

              <View style={styles.dividerRow}>
                <View style={styles.dividerLine} />
                <Text style={styles.dividerText}>Or</Text>
                <View style={styles.dividerLine} />
              </View>

              <TouchableOpacity style={styles.googleButton}>
                <Image
                  source={require('../../images/google.png')}
                  style={styles.googleIcon}
                  resizeMode="contain"
                />
                <Text style={styles.googleText}>Continue with Google</Text>
              </TouchableOpacity>
            </View>
          </View>
        </SafeAreaView>
      </ImageBackground>
    </View>
  );
};

const styles = {
  page: {
    flex: 1,
    backgroundColor: '#0b1a2d',
  },
  background: {
    flex: 1,
  },
  safeArea: {
    flex: 1,
    paddingHorizontal: 16,
    paddingTop: 16,
    paddingBottom: 20,
  },
  brandRow: {
    position: 'absolute',
    top: 180,
    left: 0,
    right: 0,
    alignItems: 'center',
    zIndex: 2,
  },
  formCardWrap: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingTop: 64,
  },
  brandText: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#ffffff',
    textAlign: 'center',
  },
  formCard: {
    backgroundColor: '#f8fbff',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 36,
    width: '100%',
    maxWidth: 380,
    shadowColor: 'rgba(12, 29, 64, 0.22)',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.35,
    shadowRadius: 20,
    elevation: 6,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#16233d',
    marginBottom: 4,
    textAlign: 'center',
  },
  errorText: {
    color: '#d14343',
    fontSize: 12,
    textAlign: 'center',
    marginBottom: 8,
  },
  signupRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    marginBottom: 14,
  },
  signupHint: {
    color: '#5b6c8c',
    fontSize: 13,
  },
  signupLink: {
    color: '#2f6dff',
    fontSize: 13,
    fontWeight: '600',
  },
  fieldBlock: {
    marginBottom: 12,
  },
  label: {
    color: '#5b6c8c',
    fontSize: 12,
    marginBottom: 6,
  },
  inputWrap: {
    position: 'relative',
    borderWidth: 1,
    borderColor: 'rgba(22, 35, 61, 0.12)',
    borderRadius: 12,
    backgroundColor: '#ffffff',
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  input: {
    color: '#16233d',
    fontSize: 14,
    paddingRight: 30,
  },
  iconRight: {
    position: 'absolute',
    right: 12,
    top: 10,
    width: 18,
    height: 18,
    tintColor: '#8a97b2',
  },
  optionsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  rememberRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  checkbox: {
    width: 16,
    height: 16,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: 'rgba(22, 35, 61, 0.25)',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#ffffff',
  },
  checkboxChecked: {
    borderColor: '#2f6dff',
  },
  checkboxInner: {
    width: 8,
    height: 8,
    borderRadius: 2,
    backgroundColor: '#2f6dff',
  },
  rememberText: {
    color: '#5b6c8c',
    fontSize: 12,
  },
  forgotText: {
    color: '#2f6dff',
    fontSize: 12,
    fontWeight: '600',
  },
  primaryButton: {
    backgroundColor: '#2f6dff',
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '700',
  },
  dividerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 12,
    gap: 10,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: 'rgba(22, 35, 61, 0.12)',
  },
  dividerText: {
    color: '#5b6c8c',
    fontSize: 12,
  },
  googleButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(22, 35, 61, 0.15)',
    paddingVertical: 10,
    gap: 8,
    backgroundColor: '#ffffff',
  },
  googleIcon: {
    width: 18,
    height: 18,
  },
  googleText: {
    color: '#5b6c8c',
    fontSize: 13,
    fontWeight: '600',
  },
} as const;

export default Login;
