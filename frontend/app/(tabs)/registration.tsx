import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import {
  Image,
  ImageBackground,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StatusBar,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { registerUser } from '../../services/auth';

const Registration = () => {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [fullName, setFullName] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const handleRegister = async () => {
    if (isLoading) return;
    setErrorMessage('');

    if (!fullName.trim() || !email.trim() || !password) {
      setErrorMessage('Please fill in all required fields.');
      return;
    }

    if (password !== confirmPassword) {
      setErrorMessage('Passwords do not match.');
      return;
    }

    setIsLoading(true);
    try {
      await registerUser({
        email,
        full_name: fullName,
        password,
      });
      router.replace('/(tabs)/camera');
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : 'Registration failed. Try again.';
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
          <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : undefined}
            style={styles.keyboardView}
          >
            <ScrollView
              contentContainerStyle={styles.scrollContent}
              keyboardShouldPersistTaps="handled"
              showsVerticalScrollIndicator={false}
            >
              <View style={styles.brandRow}>
                <Text style={styles.brandText}>Gestura</Text>
              </View>

              <View style={styles.formCard}>
                <Text style={styles.title}>Register</Text>
                <View style={styles.signupRow}>
                  <Text style={styles.signupHint}>Already have an account?</Text>
                  <TouchableOpacity onPress={() => router.push('/(tabs)/login')}>
                    <Text style={styles.signupLink}>Log In</Text>
                  </TouchableOpacity>
                </View>

                {errorMessage ? (
                  <Text style={styles.errorText}>{errorMessage}</Text>
                ) : null}

                <View style={styles.fieldBlock}>
                  <Text style={styles.label}>Full Name</Text>
                  <View style={styles.inputWrap}>
                    <TextInput
                      value={fullName}
                      onChangeText={setFullName}
                      placeholder="Lois Becket"
                      placeholderTextColor="#7f8aa4"
                      autoCapitalize="words"
                      style={styles.input}
                    />
                  </View>
                </View>

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

                <View style={styles.fieldBlock}>
                  <Text style={styles.label}>Confirm Password</Text>
                  <View style={styles.inputWrap}>
                    <TextInput
                      value={confirmPassword}
                      onChangeText={setConfirmPassword}
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

                <TouchableOpacity
                  style={styles.primaryButton}
                  onPress={handleRegister}
                >
                  <Text style={styles.primaryButtonText}>
                    {isLoading ? 'Creating...' : 'Create Account'}
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
            </ScrollView>
          </KeyboardAvoidingView>
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
  },
  keyboardView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingTop: 28,
    paddingBottom: 28,
  },
  brandRow: {
    alignItems: 'center',
    marginBottom: 22,
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
    paddingHorizontal: 18,
    paddingVertical: 24,
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
    marginBottom: 10,
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
    minHeight: 44,
    justifyContent: 'center',
  },
  input: {
    color: '#16233d',
    fontSize: 14,
    lineHeight: 18,
    paddingVertical: 0,
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

export default Registration;
