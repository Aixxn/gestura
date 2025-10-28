import { Camera, useCameraDevice, useCameraPermission, useFrameProcessor } from 'react-native-vision-camera';
import { useState, useRef, useCallback, useEffect } from 'react';
import { Image, ImageBackground, Text, TouchableOpacity, View} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { cameraStyles } from '../../constants/styles';
import * as Speech from 'expo-speech';
import { useGesturaAPI } from '../../hooks/useGesturaAPI';
import { useGesturaWebSocket } from '../../hooks/useGesturaWebSocket';

interface CapturedFrame {
  id: string;
  width: number;
  height: number;
  timestamp: number;
}

export default function CameraComponent() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [isActive, setIsActive] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isFrontCamera, setIsFrontCamera] = useState(true);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [capturedFrames, setCapturedFrames] = useState<CapturedFrame[]>([]);
  const [currentFrameInfo, setCurrentFrameInfo] = useState<{ width: number; height: number; timestamp: number } | null>(null);
  
  const device = useCameraDevice(isFrontCamera ? 'front' : 'back');
  const camera = useRef<Camera>(null);
  const frameCount = useRef(0);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // HTTP API integration (for sending frames)
  const { 
    sessionUUID, 
    sendFrame,
    stopProcessing,
    isSending,
  } = useGesturaAPI();

  // WebSocket integration (for receiving translations)
  const {
    translation,
    isConnected,
    isConnecting,
    error: wsError,
    connect: connectWebSocket,
    disconnect: disconnectWebSocket,
    clearTranslation,
  } = useGesturaWebSocket({
    uuid: sessionUUID,
    enabled: isActive, // Only connect when camera is active
    autoReconnect: true,
    reconnectInterval: 3000,
    maxReconnectAttempts: 5,
  });

  // Update translation text when received from WebSocket
  useEffect(() => {
    if (translation) {
      console.log('Translation received:', translation);
    }
  }, [translation]);

  const onError = useCallback((error: any) => {
    console.error('Camera error:', error);
    setCameraError(error.message || 'Camera error occurred');
  }, []);

  // Capture and send frames at regular intervals when active
  useEffect(() => {
    if (isActive && currentFrameInfo) {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
      
      captureIntervalRef.current = setInterval(async () => {
        if (!camera.current) return;

        try {
          // Take photo from camera
          const photo = await camera.current.takePhoto({
            flash: 'off',
            enableShutterSound: false,
          });

          // Update frame count and display
          const newFrame: CapturedFrame = {
            id: `frame_${frameCount.current++}`,
            width: currentFrameInfo.width,
            height: currentFrameInfo.height,
            timestamp: Date.now(),
          };
          
          setCapturedFrames(prev => [...prev.slice(-9), newFrame]);
          console.log(`Frame captured: ${newFrame.width}x${newFrame.height} - Total: ${frameCount.current}`);

          // Send frame to API Gateway (COMMENTED OUT FOR WEBSOCKET TESTING)
          // if (photo && sessionUUID) {
          //   // Send the file path directly
          //   await sendFrame(photo.path);
          //   console.log(`Frame sent to API with UUID: ${sessionUUID}`);
          // }
        } catch (error) {
          console.error('Failed to capture/send frame:', error);
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
  }, [isActive, currentFrameInfo, camera, sessionUUID, sendFrame]);
  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    // Store frame info for capture (runs on every frame but doesn't capture)
    const frameInfo = {
      width: frame.width,
      height: frame.height,
      timestamp: frame.timestamp
    };
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

  const handleTapToStart = async () => {
    const newActiveState = !isActive;
    setIsActive(newActiveState);
    console.log('Sign language detection active:', newActiveState);
    
    if (newActiveState) {
      setCapturedFrames([]);
      frameCount.current = 0;
      clearTranslation();
      
      // WebSocket will auto-connect via the hook when isActive becomes true
      console.log(`WebSocket will connect with UUID: ${sessionUUID}`);
      
      // Set initial frame info for immediate capture
      if (device) {
        setCurrentFrameInfo({ width: 1920, height: 1080, timestamp: Date.now() });
      }
      console.log('Started capturing and sending frames...');
    } else {
      // Stop processing and disconnect
      await stopProcessing();
      // WebSocket will auto-disconnect via the hook when isActive becomes false
      console.log(`Stopped capturing. Total frames: ${frameCount.current}`);
    }
  };

  const handleTextToSpeech = async () => {
    const textToSpeak = translation || 'Your Translation will appear here';
    
    if (isPlaying) {
      // Stop speaking if already playing
      await Speech.stop();
      setIsPlaying(false);
      console.log('Stopped speech');
    } else {
      // Start speaking
      setIsPlaying(true);
      console.log('Speaking:', textToSpeak);
      
      Speech.speak(textToSpeak, {
        language: 'en-US',
        pitch: 1.0,
        rate: 0.9,
        onDone: () => {
          setIsPlaying(false);
          console.log('Finished speaking');
        },
        onError: (error) => {
          setIsPlaying(false);
          console.error('Speech error:', error);
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
        photo={true}
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
          <View style={[
            cameraStyles.statusDisplayContainer,
            isConnected ? cameraStyles.statusDisplayConnected : cameraStyles.statusDisplayDisconnected
          ]}>
            <Text style={cameraStyles.statusDisplayTitle}>
              {isConnected ? '🟢 CONNECTED' : isConnecting ? '� CONNECTING' : '🔴 DISCONNECTED'} - Frames: {capturedFrames.length}
            </Text>
            <Text style={cameraStyles.statusDisplaySession}>
              Session: {sessionUUID.substring(0, 8)}...
            </Text>
            {wsError && (
              <Text style={[cameraStyles.statusDisplayFrame, { color: '#ff6b6b' }]}>
                Error: {wsError}
              </Text>
            )}
            {capturedFrames.length > 0 && (
              <View>
                <Text style={cameraStyles.statusDisplayFrame}>
                  Latest Frame: {capturedFrames[capturedFrames.length - 1].width}x{capturedFrames[capturedFrames.length - 1].height}
                </Text>
              </View>
            )}
          </View>
        )}
        <View style={cameraStyles.translationContainer}>
          <View style={cameraStyles.translationContent}>
            {translation ? (
              <Text style={cameraStyles.translationText}>{translation}</Text>
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
