import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import { useState, useRef, useCallback } from 'react';
import { Image, ImageBackground, Text, TouchableOpacity, View } from 'react-native';
import { cameraStyles } from './styles';

export default function CameraComponent() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [isActive, setIsActive] = useState(false);
  const [translationText, setTranslationText] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [isFrontCamera, setIsFrontCamera] = useState(true);
  
  const device = useCameraDevice(isFrontCamera ? 'front' : 'back');
  const camera = useRef<Camera>(null);

  // Frame processor callback
  const frameProcessor = useCallback((frame) => {
    'worklet';
    try {
      // TODO: Add sign language detection processing here
      console.log(`Frame: ${frame.width}x${frame.height} (${frame.pixelFormat})`);
    } catch (error) {
      console.error('Frame processing error:', error);
    }
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
    return <View style={cameraStyles.loadingContainer} />;
  }

  const handleTapToStart = () => {
    setIsActive(!isActive);
    // Toggle frame processing
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
              isActive={isActive}
              frameProcessor={frameProcessor}
              frameProcessorFps={5}
              format={{ // Add format configuration
                photoCodec: 'jpeg',
                videoCodec: 'h264',
              }}
              photo={true} // Enable photo capture
              video={false} // Disable video recording
              audio={false} // Disable audio recording
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
            <Text style={cameraStyles.tapToStartText}>Tap to Start</Text>
          </TouchableOpacity>
        </View>

      </ImageBackground>
    </View>
  );
}