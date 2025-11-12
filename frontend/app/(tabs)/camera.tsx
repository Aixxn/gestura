import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
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

interface QueuedFrame {
  id: string;
  path: string;
  timestamp: number;
}

export default function CameraComponent() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [isActive, setIsActive] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isFrontCamera, setIsFrontCamera] = useState(true);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [capturedFrames, setCapturedFrames] = useState<CapturedFrame[]>([]);
  
  const device = useCameraDevice(isFrontCamera ? 'front' : 'back');
  const camera = useRef<Camera>(null);
  const frameCount = useRef(0);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  
  // Frame queue system - max 80 frames
  const frameQueue = useRef<QueuedFrame[]>([]);
  const isProcessingQueue = useRef(false);
  const isStopping = useRef(false);
  const MAX_QUEUE_SIZE = 80;
  const CONCURRENT_UPLOADS = 5; // Process 5 frames in parallel for better throughput

  // HTTP API integration (for sending frames)
  const { 
    sessionUUID, 
    sendFrame,
    stopProcessing,
    isSending,
  } = useGesturaAPI();

  // WebSocket integration (for receiving translations)
  // Connection is persistent - only connects when UUID is ready, stays connected throughout
  const {
    translation,
    isConnected,
    isConnecting,
    error: wsError,
    clearTranslation,
  } = useGesturaWebSocket({
    uuid: sessionUUID,
    enabled: !!sessionUUID, // Only connect when UUID is ready, then stay connected
    autoReconnect: true,
    reconnectInterval: 3000,
    maxReconnectAttempts: 5,
  });

  // Monitor WebSocket connection status
  useEffect(() => {
    if (isConnected) {
      console.log('WebSocket connected successfully with UUID:', sessionUUID);
    } else if (isConnecting) {
      console.log('WebSocket connecting...');
    } else {
      console.log('WebSocket disconnected');
    }
  }, [isConnected, isConnecting, sessionUUID]);

  // Update translation text when received from WebSocket
  useEffect(() => {
    if (translation) {
      console.log('Translation received:', translation);
    }
  }, [translation]);

  // Frame queue processor - continuously sends frames from queue in parallel
  const processFrameQueue = useCallback(async () => {
    if (isProcessingQueue.current || frameQueue.current.length === 0) {
      return;
    }

    isProcessingQueue.current = true;

    while (frameQueue.current.length > 0) {
      // Take up to CONCURRENT_UPLOADS frames from queue for parallel processing
      const batch = frameQueue.current.splice(0, CONCURRENT_UPLOADS);
      
      if (batch.length === 0) break;

      console.log(`Processing batch of ${batch.length} frames in parallel (${frameQueue.current.length} remaining in queue)`);

      // Send all frames in the batch simultaneously
      const results = await Promise.allSettled(
        batch.map(async (frame) => {
          try {
            await sendFrame(frame.path);
            console.log(`✓ Frame ${frame.id} sent successfully`);
            return { success: true, frameId: frame.id };
          } catch (error) {
            console.error(`✗ Failed to send frame ${frame.id}:`, error);
            return { success: false, frameId: frame.id, error };
          }
        })
      );

      // Log batch completion summary
      const successful = results.filter(r => r.status === 'fulfilled' && r.value.success).length;
      const failed = batch.length - successful;
      console.log(`Batch complete: ${successful} successful, ${failed} failed`);
    }

    isProcessingQueue.current = false;
    
    // If we're stopping and queue is empty, complete the stop process
    if (isStopping.current && frameQueue.current.length === 0) {
      console.log('All queued frames sent. Stop process complete.');
      isStopping.current = false;
    }
  }, [sendFrame, CONCURRENT_UPLOADS]);

  // Add frame to queue (max 80 frames)
  const addFrameToQueue = useCallback((frame: QueuedFrame) => {
    // If queue is full, wait for it to have space
    // Frame will still be added - backend will process when ready
    if (frameQueue.current.length >= MAX_QUEUE_SIZE) {
      console.log(`Queue at capacity (${MAX_QUEUE_SIZE}). Frame queued, will process when backend catches up.`);
    }
    
    frameQueue.current.push(frame);
    console.log(`Frame added to queue. Queue size: ${frameQueue.current.length}/${MAX_QUEUE_SIZE}`);
    
    // Trigger queue processing
    processFrameQueue();
    return true; // Always indicate success
  }, [processFrameQueue, MAX_QUEUE_SIZE]);

  const onError = useCallback((error: any) => {
    console.error('Camera error:', error);
    setCameraError(error.message || 'Camera error occurred');
  }, []);

  // Capture and send frames at regular intervals when active
  useEffect(() => {
    if (isActive) {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
      
      captureIntervalRef.current = setInterval(async () => {
        if (!camera.current || isStopping.current) return;

        try {
          // Take photo from camera
          const photo = await camera.current.takePhoto({
            flash: 'off',
            enableShutterSound: false,
          });

          // Update frame count and display
          const newFrame: CapturedFrame = {
            id: `frame_${frameCount.current}`,
            width: 1920, // Default camera resolution
            height: 1080,
            timestamp: Date.now(),
          };
          
          frameCount.current++;
          setCapturedFrames(prev => [...prev.slice(-9), newFrame]);
          console.log(`Frame captured: ${newFrame.width}x${newFrame.height} - Total: ${frameCount.current}`);

          // Add frame to queue instead of sending directly
          if (photo && sessionUUID) {
            const queuedFrame: QueuedFrame = {
              id: newFrame.id,
              path: photo.path,
              timestamp: newFrame.timestamp,
            };
            addFrameToQueue(queuedFrame);
          }
        } catch (error) {
          console.error('Failed to capture frame:', error);
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
  }, [isActive, camera, sessionUUID, addFrameToQueue]);

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
      // Starting detection
      isStopping.current = false;
      setCapturedFrames([]);
      frameCount.current = 0;
      frameQueue.current = []; // Clear any existing queue
      clearTranslation();
      
      console.log(`WebSocket will connect with UUID: ${sessionUUID}`);
      console.log('Started capturing and sending frames...');
    } else {
      // Stopping detection - process remaining frames in queue
      console.log(`Stopping capture. Processing ${frameQueue.current.length} remaining frames in queue...`);
      isStopping.current = true;
      
      // Wait for queue to be fully processed
      const checkQueueEmpty = setInterval(() => {
        if (frameQueue.current.length === 0 && !isProcessingQueue.current) {
          clearInterval(checkQueueEmpty);
          console.log(`All frames processed. Total frames captured: ${frameCount.current}`);
          
          // Now stop the backend processing
          stopProcessing().then(() => {
            console.log('Backend processing stopped');
          });
        } else {
          console.log(`Waiting for queue to empty... ${frameQueue.current.length} frames remaining`);
        }
      }, 500);
      
      // No timeout - will process ALL frames no matter how long it takes
      console.log('Will process all frames without timeout. Queue will be fully flushed.');
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
              {isConnected ? 'CONNECTED' : isConnecting ? 'CONNECTING' : 'DISCONNECTED'} - Frames: {frameCount.current}
            </Text>
            <Text style={cameraStyles.statusDisplaySession}>
              Session: {sessionUUID.substring(0, 8)}... | Queue: {frameQueue.current.length}/{MAX_QUEUE_SIZE}
            </Text>
            {wsError && (
              <Text style={[cameraStyles.statusDisplayFrame, { color: '#ff6b6b' }]}>
                Error: {wsError}
              </Text>
            )}
            {isStopping.current && (
              <Text style={[cameraStyles.statusDisplayFrame, { color: '#ffd700' }]}>
                Stopping... Processing {frameQueue.current.length} remaining frames
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
