import { useRouter } from 'expo-router';
import React from 'react';
import { ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

// --- Color Palette ---
const COLORS = {
  primaryBlue: '#ffffffff', 
  accentBlue: '#0917b4ff', 
  background: '#0a2c8aff', 
  textPrimary: '#ffffffff', 
  textSecondary: '#929bf4ff',
  white: '#FFFFFF',
  lightBlueCircle: 'rgba(0, 122, 255, 0.15)', 
  dotInactive: '#D1D5DB',
};

// --- Styles ---
// React Native StyleSheet
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  topBar: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 10,
    height: 50,
    position: 'relative',
  },
  logoContainer: {
    flex: 1,
    alignItems: 'center',
  },
  logoIcon: {
    fontSize: 28,
  },
  closeButton: {
    position: 'absolute',
    right: 20,
    top: 10,
    padding: 5,
  },
  closeIcon: {
    fontSize: 32,
    color: COLORS.textSecondary,
    fontWeight: '300',
  },
  content: {
    flex: 1,
    alignItems: 'center',
    paddingHorizontal: 30,
    paddingVertical: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: COLORS.textPrimary,
    textAlign: 'center',
    marginTop: 250,
    marginBottom: 0,
  },
  subtitle: {
    fontSize: 16,
    color: COLORS.textSecondary,
    textAlign: 'center',
    marginTop: 15,
    lineHeight: 24,
    maxWidth: '90%',
  },
  dotsContainer: {
    flexDirection: 'row',
    marginTop: 40,
  },
  dot: {
    marginTop: 100,
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: COLORS.dotInactive,
    marginHorizontal: 4,
  },
  dotActive: {
    backgroundColor: COLORS.white,
  },
  bottomActions: {
    paddingHorizontal: 30,
    paddingVertical: 20,
    paddingBottom: 40,
  },
  primaryButton: {
    backgroundColor: COLORS.accentBlue,
    paddingVertical: 18,
    paddingHorizontal: 18,
    borderRadius: 30,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  primaryButtonText: {
    fontSize: 18,
    color: COLORS.textPrimary,
    fontWeight: 'bold',
  },
  secondaryButton: {
    marginTop: 20,
    alignItems: 'center',
  },
  secondaryButtonText: {
    fontSize: 16,
    color: COLORS.textSecondary,
    fontWeight: '500',
  },
});


const GesturaLastPage = () => {
  const router = useRouter();

  const handleGetStarted = () => {
    router.push('/(tabs)/camera');
  };

  return (
    <ScrollView style={styles.container}>
      {/* --- Top Bar: Logo and Close Button --- */}
      <View style={styles.topBar}>
        <View style={styles.logoContainer}>
          <Text style={styles.logoIcon}></Text>
        </View>
        <TouchableOpacity style={styles.closeButton}>
          <Text style={styles.closeIcon}>×</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.content}>
        {/* --- Welcome Text --- */}
        <Text style={styles.title}>
        Gestura <Text></Text>
        </Text>
        <Text style={styles.subtitle}>
            Thank you for trying Gestura. We hope it enhances your communication experience.
        </Text>       
        {/* --- Page Indicator Dots --- */}
        <View style={styles.dotsContainer}>
          <View style={styles.dot} />
          <View style={styles.dot} />
          <View style={[styles.dot, styles.dotActive]} />
        </View>
      </View>

      {/* --- Bottom Buttons --- */}
      <View style={styles.bottomActions}>
        <TouchableOpacity style={styles.primaryButton} onPress={handleGetStarted}>
          <Text style={styles.primaryButtonText}>Continue</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

export default GesturaLastPage;

