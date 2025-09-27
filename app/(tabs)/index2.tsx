import { useRouter } from 'expo-router';
import React from 'react';
import { ImageBackground, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

// --- Color Palette ---
// Derived from your "Gestura" system image and "ASL Bloom" for the accent color.
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
  },
  backgroundImage: {
    flex:1,
    width: '100%',
    height: '100%',
  },
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(10, 44, 138, 0.14)', // Semi-transparent overlay to maintain text readability
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
    alignItems: 'center',
    paddingHorizontal: 30,
    paddingVertical: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: COLORS.textPrimary,
    textAlign: 'center',
    marginTop: 130,
    marginBottom: 0,
  },
  subtitle: {
    fontSize: 16,
    color: COLORS.textSecondary,
    textAlign: 'center',
    marginTop: 15,
    marginBottom: 180,
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


const GesturaLandingPage = () => {
  const router = useRouter();

  const handleGetStarted = () => {
    router.push('/(tabs)/index3');
  };

  return (
    <ImageBackground 
      source={require('../../images/background.png')} 
      style={styles.backgroundImage}
      resizeMode="cover"
    >
      <View style={styles.overlay}>
        <View style={styles.container}>
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
              Welcome to Gestura <Text></Text>
            </Text>
            <Text style={styles.subtitle}>
              This system is operated by three Computer Science Students.
            </Text>       
            {/* --- Page Indicator Dots --- */}
            <View style={styles.dotsContainer}>
              <View style={styles.dot} />
              <View style={[styles.dot, styles.dotActive]} />
              <View style={styles.dot} />
            </View>
          </View>

          {/* --- Bottom Buttons --- */}
          <View style={styles.bottomActions}>
            <TouchableOpacity style={styles.primaryButton} onPress={handleGetStarted}>
              <Text style={styles.primaryButtonText}>Next</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </ImageBackground>
  );
};

export default GesturaLandingPage;

