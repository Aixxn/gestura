import { Camera, useCameraDevice, useCameraPermission, useFrameProcessor } from 'react-native-vision-camera';
import { useState, useRef, useCallback } from 'react';
import { Image, ImageBackground, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { cameraStyles } from '../../constants/styles';

export default function CameraComponent() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [isActive, setIsActive] = useState(false);
  const [translationText, setTranslationText] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [isFrontCamera, setIsFrontCamera] = useState(true);
  const [cameraError, setCameraError] = useState<string | null>(null);
  
  const device = useCameraDevice(isFrontCamera ? 'front' : 'back');
  const camera = useRef<Camera>(null);

  // Camera error handler
  const onError = useCallback((error: any) => {
    console.error('Camera error:', error);
    setCameraError(error.message || 'Camera error occurred');
  }, []);

  // Process frame on JS thread (for API calls, state updates, etc.)
  const processFrameOnJS = useCallback(async (width: number, height: number, timestamp: number) => {
    try {
      // TODO: Send frame to your backend for sign language detection
      console.log(`Processing frame: ${width}x${height} @ ${timestamp}ms`);
      
      // Example API call to your backend
      // const response = await fetch('http://your-backend-url/detect', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ width, height, timestamp }),
      // });
      // const result = await response.json();
      // setTranslationText(result.detectedSign);
      
    } catch (error) {
      console.error('Frame processing error:', error);
    }
  }, []);

  // Frame processor - runs on native thread for performance
  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    
    // Access frame properties (don't use console.log in worklets - it causes crashes)
    // const width = frame.width;
    // const height = frame.height;
    // const timestamp = frame.timestamp;
    
    // TODO: Process frame for sign language detection
    // You can use ML models here or send to backend
    
  }, []);

  const toggleCameraFacing = useCallback(() => {
    setIsFrontCamera(!isFrontCamera);
  }, [isFrontCamera]);

  if (!hasPermission) {
    return (
      <View style={cameraStyles.permissionContainer}>
        <Text style={cameraStyles.permissionMessage}>We need your permission to show the camera</Text>
        <TouchableOpacity style={cameraStyles.permissionButton} onPress={requestPermission}>
          <Text style={cameraStyles.permissionButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={cameraStyles.loadingContainer}>
        <Text style={cameraStyles.permissionMessage}>Loading camera...</Text>
      </View>
    );
  }

  if (cameraError) {
    return (
      <View style={cameraStyles.permissionContainer}>
        <Text style={cameraStyles.permissionMessage}>Camera Error: {cameraError}</Text>
        <TouchableOpacity style={cameraStyles.permissionButton} onPress={() => setCameraError(null)}>
          <Text style={cameraStyles.permissionButtonText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const handleTapToStart = () => {
    setIsActive(!isActive);
    console.log('Camera active:', !isActive);
  };

  const handlePlayAudio = () => {
    setIsPlaying(!isPlaying);
    // Add text-to-speech functionality here when ready
    console.log('Playing audio for:', translationText);
  };

  const handleTranslatePress = () => {
    console.log('Translate functionality');
    // Add translate functionality here
  };

  return (
    <SafeAreaView style={{ flex: 1 }}>
      <View style={cameraStyles.container}>
        <ImageBackground 
          source={require('../../images/Camera-bg.png')} 
          style={cameraStyles.backgroundImage}
          resizeMode="cover"
        >
        {/* Camera Container with border alignment */}
        <View style={cameraStyles.cameraContainer}>
          <View style={cameraStyles.cameraFrame}>
            <Camera
              ref={camera}
              style={cameraStyles.camera}
              device={device}
              isActive={true}
              onError={onError}
              photo={true}
              video={false}
              audio={false}
            />
          </View>
        </View>
        
        {/* Top Status Bar with Camera Flip Button */}
        <View style={cameraStyles.topOverlay}>
          <Text style={cameraStyles.statusText}>Gestura</Text>
          <TouchableOpacity style={cameraStyles.cameraFlipButton} onPress={toggleCameraFacing}>
            <Image
              source={require('../../images/camera.png')}
              style={cameraStyles.cameraFlipIcon}
              resizeMode="contain"
            />
          </TouchableOpacity>
        </View>

        {/* Translation Interface */}
        <View style={cameraStyles.translationContainer}>
          <View style={cameraStyles.translationContent}>
            {translationText ? (
              <Text style={cameraStyles.translationText}>{translationText}</Text>
            ) : (
              <Text style={cameraStyles.placeholderText}>Your Translation will appear here...</Text>
            )}
          </View>
          
          {/* Audio Control */}
          <TouchableOpacity 
            style={[cameraStyles.audioButton, isPlaying && cameraStyles.audioButtonActive]} 
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
        <View style={cameraStyles.bottomActionsContainer}>
          <TouchableOpacity 
            style={cameraStyles.actionButton} 
            onPress={handleTranslatePress}
          >
            <Image
              source={require('../../images/translate.png')}
              style={cameraStyles.actionButtonImage}
              resizeMode="contain"
            />
          </TouchableOpacity>

          <TouchableOpacity 
            style={cameraStyles.tapToStartButton} 
            onPress={handleTapToStart}
          >
            <Image
              source={require('../../images/taptostart.png')}
              style={cameraStyles.tapToStartIcon}
              resizeMode="contain"
            />
            <Text style={cameraStyles.tapToStartText}>
              {isActive ? 'Tap to Stop' : 'Tap to Start'}
            </Text>
          </TouchableOpacity>
        </View>

        </ImageBackground>
      </View>
    </SafeAreaView>
  );
}
