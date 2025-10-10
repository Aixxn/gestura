import { CameraType, CameraView, useCameraPermissions } from 'expo-camera';
import { useState } from 'react';
import { Image, ImageBackground, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

export default function CameraComponent() {
  const [facing, setFacing] = useState<CameraType>('front');
  const [permission, requestPermission] = useCameraPermissions();
  const [translationText, setTranslationText] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);

  if (!permission) {
    return <View style={styles.loadingContainer} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionMessage}>We need your permission to show the camera</Text>
        <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
          <Text style={styles.permissionButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  function toggleCameraFacing() {
    setFacing(current => (current === 'front' ? 'back' : 'front'));
  }

  const handlePlayAudio = () => {
    setIsPlaying(!isPlaying);
    // Add text-to-speech functionality here when ready
    console.log('Playing audio for:', translationText);
  };

  const handleTapToStart = () => {
    console.log('Tap to start functionality');
    // Add tap to start functionality here
  };

  const handleTranslatePress = () => {
    console.log('Translate functionality');
    // Add translate functionality here
  };

  return (
    <View style={styles.container}>
      <ImageBackground 
        source={require('../../images/Camera-bg.png')} 
        style={styles.backgroundImage}
        resizeMode="cover"
      >
        {/* Camera Container with border alignment */}
        <View style={styles.cameraContainer}>
          <View style={styles.cameraFrame}>
            <CameraView style={styles.camera} facing={facing} />
          </View>
        </View>
        
        {/* Top Status Bar with Camera Flip Button */}
        <View style={styles.topOverlay}>
          <Text style={styles.statusText}>Gestura</Text>
          <TouchableOpacity style={styles.cameraFlipButton} onPress={toggleCameraFacing}>
            <Image
              source={require('../../images/camera.png')}
              style={styles.cameraFlipIcon}
              resizeMode="contain"
            />
          </TouchableOpacity>
        </View>

        {/* Translation Interface */}
        <View style={styles.translationContainer}>
          <View style={styles.translationContent}>
            {translationText ? (
              <Text style={styles.translationText}>{translationText}</Text>
            ) : (
              <Text style={styles.placeholderText}>Your Translation will appear here...</Text>
            )}
          </View>
          
          {/* Audio Control */}
          <TouchableOpacity 
            style={[styles.audioButton, isPlaying && styles.audioButtonActive]} 
            onPress={handlePlayAudio}
            disabled={!translationText}
          >
            <Image
              source={require('../../images/volume.png')}
              style={{ width: 20, height: 20, tintColor: isPlaying ? 'white' : '#000000ff' }}
              resizeMode="contain"
            />
          </TouchableOpacity>
        </View>

        {/* Bottom Action Buttons */}
        <View style={styles.bottomActionsContainer}>
          <TouchableOpacity style={styles.actionButton} onPress={handleTranslatePress}>
            <Image
              source={require('../../images/translate.png')}
              style={styles.actionButtonImage}
              resizeMode="contain"
            />
          </TouchableOpacity>

          <TouchableOpacity style={styles.tapToStartButton} onPress={handleTapToStart}>
            <Image
              source={require('../../images/taptostart.png')}
              style={styles.tapToStartImage}
              resizeMode="contain"
            />
          </TouchableOpacity>
        </View>

      </ImageBackground>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  loadingContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  permissionContainer: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 30,
  },
  permissionMessage: {
    textAlign: 'center',
    paddingBottom: 30,
    color: 'white',
    fontSize: 18,
    lineHeight: 26,
  },
  permissionButton: {
    backgroundColor: '#0917b4',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 25,
  },
  permissionButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  backgroundImage: {
    flex: 1,
    width: '100%',
    height: '100%',
  },
  cameraContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cameraFrame: {
    width: '100%',
    marginTop: 70,
    height: 700, // Fixed height instead of aspectRatio
    overflow: 'hidden',
  },
  camera: {
    flex: 1,
    width: '100%',
    height: '100%',
  },
  topOverlay: {
    position: 'absolute',
    top: 60,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    paddingHorizontal: 20,
    backgroundColor: 'transparent',
  },
  statusText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    flex: 1,
    textAlign: 'center',
  },
  cameraFlipButton: {
    position: 'absolute',
    right: 20,
    padding: 10,
  },
  cameraFlipIcon: {
    width: 24,
    height: 24,
    tintColor: 'white',
  },
  translationContainer: {
    position: 'absolute',
    bottom: 37,
    left: 0,
    right: 0,
    height: 200,
    opacity: 0.65,
    backgroundColor: 'rgba(255, 255, 255, 1)',
    padding: 25,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 5,
  },
  translationContent: {
    flexGrow: 1,
  },
  translationText: {
    fontSize: 16,
    lineHeight: 22,
    color: '#333',
    textAlign: 'left',
  },
  placeholderText: {
    fontSize: 14,
    color: '#999',
    textAlign: 'left',
    fontStyle: 'italic',
  },
  audioButton: {
    position: 'absolute',
    top: 15,
    right: 15,
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#f0f0f0',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  audioButtonActive: {
    backgroundColor: '#0917b4',
  },
  bottomActionsContainer: {
    position: 'absolute',
    bottom: -20,
    left: 30,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 60,
  },
  actionButton: {
    padding: 10,
  },
  actionButtonImage: {
    width: 50,
    height: 50,
  },
  tapToStartButton: {
    padding: 15,
  },
  tapToStartImage: {
    width: 130,
    height: 100,
  },
});