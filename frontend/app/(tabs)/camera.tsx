import { Camera, useCameraDevice, useCameraPermission, useFrameProcessor } from 'react-native-vision-camera';
import { useState, useRef, useCallback, useEffect } from 'react';
import { Image, ImageBackground, Text, TouchableOpacity, View, FlatList } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { cameraStyles } from '../../constants/styles';
import * as Speech from 'expo-speech';

interface CapturedFrame {
  id: string;
  width: number;
  height: number;
  timestamp: number;
}

export default function CameraComponent() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [isActive, setIsActive] = useState(false);
  const [translationText, setTranslationText] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [isFrontCamera, setIsFrontCamera] = useState(true);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [capturedFrames, setCapturedFrames] = useState<CapturedFrame[]>([]);
  const [currentFrameInfo, setCurrentFrameInfo] = useState<{ width: number; height: number; timestamp: number } | null>(null);
  const device = useCameraDevice(isFrontCamera ? 'front' : 'back');
  const camera = useRef<Camera>(null);
  const frameCount = useRef(0);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const onError = useCallback((error: any) => {
    console.error('Camera error:', error);
    setCameraError(error.message || 'Camera error occurred');
  }, []);

  // Capture frames at regular intervals when active
  useEffect(() => {
    if (isActive && currentFrameInfo) {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
      
      captureIntervalRef.current = setInterval(() => {
        if (currentFrameInfo) {
          const newFrame: CapturedFrame = {
            id: `frame_${frameCount.current++}`,
            width: currentFrameInfo.width,
            height: currentFrameInfo.height,
            timestamp: Date.now(),
          };
          
          setCapturedFrames(prev => [...prev.slice(-9), newFrame]);
          console.log(`✅ Frame captured: ${newFrame.width}x${newFrame.height} @ ${newFrame.timestamp}ms - Total: ${frameCount.current}`);
          
          // TODO: Send frame data to your sign language detection backend
        }
      }, 500);
    } else {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }
    }
    
    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
    };
  }, [isActive, currentFrameInfo]);

  // Frame processor - just updates frame info
  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    // Store frame info for capture (runs on every frame but doesn't capture)
    const frameInfo = {
      width: frame.width,
      height: frame.height,
      timestamp: frame.timestamp
    };
    // Note: We can't directly call setCurrentFrameInfo here
    // So we'll capture using intervals instead
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
    const newActiveState = !isActive;
    setIsActive(newActiveState);
    console.log('Sign language detection active:', newActiveState);
    
    if (newActiveState) {
      setCapturedFrames([]);
      frameCount.current = 0;
      // Set initial frame info for immediate capture
      if (device) {
        setCurrentFrameInfo({ width: 1920, height: 1080, timestamp: Date.now() });
      }
      console.log('Started capturing frames...');
    } else {
      console.log('Stopped capturing. Total frames:', frameCount.current);
    }
  };

  const handleTextToSpeech = async () => {
    const textToSpeak = translationText || 'Your Translation will appear here';
    
    if (isPlaying) {
      // Stop speaking if already playing
      await Speech.stop();
      setIsPlaying(false);
      console.log('🔇 Stopped speech');
    } else {
      // Start speaking
      setIsPlaying(true);
      console.log('🔊 Speaking:', textToSpeak);
      
      Speech.speak(textToSpeak, {
        language: 'en-US',
        pitch: 1.0,
        rate: 0.9,
        onDone: () => {
          setIsPlaying(false);
          console.log('✅ Finished speaking');
        },
        onError: (error) => {
          setIsPlaying(false);
          console.error('❌ Speech error:', error);
        },
      });
    }
  };

  const handleTranslatePress = () => {
    console.log('Translate functionality');
  };

  return (
    <SafeAreaView style={{ flex: 1 }}>
      <View style={cameraStyles.container}>
        <ImageBackground 
          source={require('../../images/Camera-bg.png')} 
          style={cameraStyles.backgroundImage}
          resizeMode="cover"
        >
        <View style={cameraStyles.cameraContainer}>
          <View style={cameraStyles.cameraFrame}>
        <Camera
        ref={camera}
        style={{ flex: 1 }}
        device={device}
        isActive={true}                
        video={true}
        audio={false}
        frameProcessor={frameProcessor}
        onError={onError}
/>

          
          </View>
        </View>
        
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

        {/* Frame Processing Status Display */}
        {isActive && (
          <View style={{ 
            position: 'absolute', 
            top: 80, 
            left: 10, 
            right: 10,
            backgroundColor: 'rgba(0,0,0,0.8)',
            padding: 10,
            borderRadius: 10,
            borderWidth: 2,
            borderColor: '#00ff00'
          }}>
            <Text style={{ 
              color: '#00ff00', 
              fontSize: 16, 
              fontWeight: 'bold',
              marginBottom: 5 
            }}>
              RECORDING - Frames Captured: {capturedFrames.length}
            </Text>
            {capturedFrames.length > 0 && (
              <View>
                <Text style={{ color: 'white', fontSize: 12 }}>
                  Latest Frame: {capturedFrames[capturedFrames.length - 1].width}x{capturedFrames[capturedFrames.length - 1].height}
                </Text>
                <Text style={{ color: 'white', fontSize: 10 }}>
                  Timestamp: {capturedFrames[capturedFrames.length - 1].timestamp}ms
                </Text>
              </View>
            )}
          </View>
        )}
        
        {/* Frame List Display */}
        {capturedFrames.length > 0 && (
          <View style={{ position: 'absolute', bottom: 180, left: 10, right: 10, height: 110 }}>
            <Text style={{ 
              color: 'white', 
              marginBottom: 5, 
              fontSize: 14,
              fontWeight: 'bold',
              backgroundColor: 'rgba(0,0,0,0.7)',
              padding: 5,
              borderRadius: 5
            }}>
              Captured Frames ({capturedFrames.length}/10)
            </Text>
            <FlatList
              horizontal
              data={capturedFrames}
              keyExtractor={(item) => item.id}
              showsHorizontalScrollIndicator={false}
              renderItem={({ item, index }) => (
                <View style={{ 
                  width: 85, 
                  height: 70, 
                  marginHorizontal: 3, 
                  backgroundColor: 'rgba(34,139,34,0.9)', 
                  borderRadius: 8,
                  justifyContent: 'center',
                  alignItems: 'center',
                  borderWidth: 2,
                  borderColor: '#00ff00'
                }}>
                  <Text style={{ color: 'white', fontSize: 12, fontWeight: 'bold' }}>
                    Frame {index + 1}
                  </Text>
                  <Text style={{ color: 'white', fontSize: 9 }}>
                    {item.width}x{item.height}
                  </Text>
                  <Text style={{ color: 'white', fontSize: 8 }}>
                    {item.timestamp}ms
                  </Text>
                </View>
              )}
            />
          </View>
        )}

        <View style={cameraStyles.translationContainer}>
          <View style={cameraStyles.translationContent}>
            {translationText ? (
              <Text style={cameraStyles.translationText}>{translationText}</Text>
            ) : (
              <Text style={cameraStyles.placeholderText}>Your Translation will appear here...</Text>
            )}
          </View>
          
          <TouchableOpacity 
            style={[cameraStyles.audioButton, isPlaying && cameraStyles.audioButtonActive]} 
            onPress={handleTextToSpeech}
          >
            <Image
              source={require('../../images/volume.png')}
              style={{ width: 20, height: 20, tintColor: isPlaying ? 'white' : '#000000ff' }}
              resizeMode="contain"
            />
          </TouchableOpacity>
        </View>

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
