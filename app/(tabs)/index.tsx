import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { ImageBackground, Text, TouchableOpacity, View } from 'react-native';
import { styles } from './styles';

const Gestura = () => {
  const router = useRouter();
  const [currentScreen, setCurrentScreen] = useState(1); 

  const handleNext = () => {
    if (currentScreen < 3) {
      setCurrentScreen(currentScreen + 1);
    } else {
      // Navigate to camera on final screen
      router.push('/(tabs)/camera');
    }
  };

  const getScreenData = () => {
    switch (currentScreen) {
      case 1:
        return {
          title: "Welcome to Gestura",
          subtitle: "Real-time conversations that matter, bringing you closer with every sign.",
          buttonText: "Get started",
          backgroundImage: require('../../images/background.png'),
          activeDot: 1
        };
      case 2:
        return {
          title: "Welcome to Gestura",
          subtitle: "This system is operated by three Computer Science Students.",
          buttonText: "Next",
          backgroundImage: require('../../images/background2.png'),
          activeDot: 2
        };
      case 3:
        return {
          title: "Welcome to Gestura",
          subtitle: "Thank you for trying Gestura. We hope it enhances your communication experience.",
          buttonText: "Start Using Gestura",
          backgroundImage: require('../../images/background3.png'),
          activeDot: 3
        };
      default:
        return {
          title: "Welcome to Gestura",
          subtitle: "Real-time conversations that matter, bringing you closer with every sign.",
          buttonText: "Get started",
          backgroundImage: require('../../images/background.png'),
          activeDot: 1
        };
    }
  };

  const screenData = getScreenData();

  return (
    <ImageBackground 
      source={screenData.backgroundImage}
      style={styles.backgroundImage}
      resizeMode="cover"
    >
      <View style={styles.overlay}>
        <View style={styles.container}>

          <View style={styles.topBar}>
            <View style={styles.logoContainer}>
              <Text style={styles.logoIcon}></Text>
            </View>
            <TouchableOpacity style={styles.closeButton}>
              <Text style={styles.closeIcon}>×</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.content}>
            {/* --- Welcome Text with Gradient Background --- */}
            <View style={styles.titleContainer}>
              
                <Text style={styles.title}>
                  {screenData.title}
                </Text>
            </View>
            <Text style={styles.subtitle}>
              {screenData.subtitle}
            </Text>       
            
            <View style={styles.dotsContainer}>
              <View style={[styles.dot, screenData.activeDot === 1 && styles.dotActive]} />
              <View style={[styles.dot, screenData.activeDot === 2 && styles.dotActive]} />
              <View style={[styles.dot, screenData.activeDot === 3 && styles.dotActive]} />
            </View>
          </View>

          {/* --- Bottom Buttons --- */}
          <View style={styles.bottomActions}>
            <TouchableOpacity style={styles.primaryButton} onPress={handleNext}>
              <Text style={styles.primaryButtonText}>{screenData.buttonText}</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </ImageBackground>
  );
};
export default Gestura;

