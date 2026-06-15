import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import { useState, useRef, useCallback, useEffect } from 'react';
import { Image, Text, TouchableOpacity, View} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { cameraStyles } from '../../constants/styles';
import * as Speech from 'expo-speech';
import * as Haptics from 'expo-haptics';
import { useGesturaAPI } from '../../hooks/useGesturaAPI';
import { useGesturaWebSocket } from '../../hooks/useGesturaWebSocket';
import { getToken } from '../../services/token';

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
  const [captureStatus, setCaptureStatus] = useState<string | null>(null);
  
  const device = useCameraDevice(isFrontCamera ? 'front' : 'back');
  const camera = useRef<Camera>(null);
  const frameCount = useRef(0);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  
  // Frame queue system - max 80 frames
  const frameQueue = useRef<QueuedFrame[]>([]);
  const isProcessingQueue = useRef(false);
  const isStopping = useRef(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const MAX_QUEUE_SIZE = 80;
  const CONCURRENT_UPLOADS = 5; // Process 5 frames in parallel for better throughput

  // HTTP API integration (for sending frames)
  const { 
    sessionUUID, 
    sendFrame,
    stopProcessing,
    isSending,
    lastStatus,
    clearSession,
    error: apiError,
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

  // Update translation text when received from WebSocket and auto-speak
  useEffect(() => {
    if (translation) {
      console.log('Translation received:', translation);
      
      // Automatically speak new translation
      (async () => {
        try {
          await Speech.stop(); // Stop any existing speech
          Speech.speak(translation, {
            language: 'en-US',
            pitch: 1.0,
            rate: 0.9,
            onDone: () => {
              console.log('Auto-TTS finished speaking:', translation);
            },
            onError: (error) => {
              console.error('Auto-TTS error:', error);
            },
          });
          console.log('Auto-TTS started speaking:', translation);
        } catch (error) {
          console.error('Failed to start auto-TTS:', error);
        }
      })();
    }
  }, [translation]);

  // Provide haptic feedback for connection status changes
  useEffect(() => {
    if (isConnected && isActive) {
      // Connection established during active session
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } else if (!isConnected && isActive && !isConnecting) {
      // Connection lost during active session
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    }
  }, [isConnected, isActive, isConnecting]);

  // Frame queue processor - continuously sends frames from queue in parallel
  const processFrameQueue = useCallback(async () => {
    if (isProcessingQueue.current || frameQueue.current.length === 0) {
      return;
    }

    isProcessingQueue.current = true;
    setCaptureStatus('Uploading frames...');

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
    setCaptureStatus(isStopping.current ? 'Finalizing translation...' : null);
    
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
        if (!camera.current || isStopping.current || !sessionUUID) return;

        try {
          // Take photo from camera
          const photo = await camera.current.takePhoto({
            flash: 'off',
            enableShutterSound: false,
          });

          const frameId = `frame_${frameCount.current}`;
          const timestamp = Date.now();
          
          frameCount.current++;
          console.log(`Frame captured: ${frameId} - Total: ${frameCount.current}`);

          // Add frame to queue instead of sending directly
          if (photo && sessionUUID) {
            const queuedFrame: QueuedFrame = {
              id: frameId,
              path: photo.path,
              timestamp,
            };
            addFrameToQueue(queuedFrame);
          }
        } catch (error) {
          console.error('Failed to capture frame:', error);
          setCaptureStatus('Failed to capture frame');
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
      <View 
        style={cameraStyles.permissionContainer}
        accessible={true}
        accessibilityRole="alert"
        accessibilityLabel="Camera permission required to use Gestura sign language translator"
      >
        <Text 
          style={cameraStyles.permissionMessage}
          accessible={true}
          accessibilityRole="header"
        >
          We need your permission to show the camera
        </Text>
        <TouchableOpacity 
          style={cameraStyles.permissionButton} 
          onPress={requestPermission}
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel="Grant camera permission"
          accessibilityHint="Activates to open camera settings and allow camera access"
          hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
        >
          <Text style={cameraStyles.permissionButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!device) {
    return (
      <View 
        style={cameraStyles.loadingContainer}
        accessible={true}
        accessibilityRole="progressbar"
        accessibilityLabel="Loading camera"
        accessibilityHint="Please wait while the camera initializes"
      >
        <Text style={cameraStyles.permissionMessage}>Loading camera...</Text>
      </View>
    );
  }

  if (cameraError) {
    return (
      <View 
        style={cameraStyles.permissionContainer}
        accessible={true}
        accessibilityRole="alert"
        accessibilityLabel={`Camera error: ${cameraError}`}
      >
        <Text 
          style={cameraStyles.permissionMessage}
          accessible={true}
          accessibilityRole="text"
        >
          Camera Error: {cameraError}
        </Text>
        <TouchableOpacity 
          style={cameraStyles.permissionButton} 
          onPress={() => {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
            setCameraError(null);
          }}
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel="Retry camera initialization"
          accessibilityHint="Activates to attempt reconnecting to camera"
          hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
        >
          <Text style={cameraStyles.permissionButtonText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const handleTapToStart = async () => {
    const newActiveState = !isActive;
    const token = getToken();

    if (newActiveState && !token) {
      setCameraError('Please log in before starting translation.');
      await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      return;
    }

    if (newActiveState && !sessionUUID) {
      setCameraError('Camera session is still initializing. Try again in a moment.');
      await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
      return;
    }

    // Provide haptic feedback based on action
    if (newActiveState) {
      await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } else {
      await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
    }
    
    setIsActive(newActiveState);
    console.log('Sign language detection active:', newActiveState);
    
    if (newActiveState) {
      // Starting detection
      isStopping.current = false;
      setCaptureStatus('Starting capture...');
      frameCount.current = 0;
      frameQueue.current = []; // Clear any existing queue
      clearTranslation();
      clearSession();
      
      console.log(`WebSocket will connect with UUID: ${sessionUUID}`);
      console.log('Started capturing and sending frames...');
    } else {
      // Stopping detection - process remaining frames in queue
      console.log(`Stopping capture. Processing ${frameQueue.current.length} remaining frames in queue...`);
      isStopping.current = true;
      setIsTranslating(true);
      setCaptureStatus('Stopping capture...');
      
      // Wait for queue to be fully processed
      const checkQueueEmpty = setInterval(async () => {
        if (frameQueue.current.length === 0 && !isProcessingQueue.current) {
          clearInterval(checkQueueEmpty);
          console.log(`All frames processed. Total frames captured: ${frameCount.current}`);
          
          // Now stop the backend processing
          const result = await stopProcessing();
          setIsTranslating(false);
          if (result.success) {
            setCaptureStatus(null);
            console.log('Backend processing stopped');
          } else {
            setCaptureStatus(result.error || 'Failed to stop backend processing');
          }
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

  const isFinishingSession = captureStatus === 'Stopping capture...' || captureStatus === 'Finalizing translation...';
  const connectionLabel = isConnected ? 'Connected' : isConnecting ? 'Connecting' : 'Offline';
  const sessionStatus = isFinishingSession
    ? 'Processing...'
    : isActive
      ? 'Listening...'
      : translation
        ? 'Translation ready'
        : 'Ready to translate';
  const sessionHint = isActive
    ? 'Keep signing clearly inside the guide'
    : translation
      ? 'Tap Start to translate again'
      : 'Place your hands inside the frame';
  const actionLabel = isFinishingSession ? 'Finishing...' : isActive ? 'Stop' : 'Start';
  const actionAccessibilityLabel = isActive ? 'Stop translation' : 'Start translation';

  return (
    <SafeAreaView style={{ flex: 1 }}>
      <View style={cameraStyles.container}>
        <View style={cameraStyles.topOverlay}>
          <View>
            <Text style={cameraStyles.statusText}>Gestura</Text>
            <Text style={cameraStyles.subtitleText}>Live sign translator</Text>
          </View>
          <View style={cameraStyles.topActions}>
            <View
              style={[
                cameraStyles.connectionPill,
                isConnected ? cameraStyles.connectionPillConnected : cameraStyles.connectionPillOffline,
              ]}
            >
              <View
                style={[
                  cameraStyles.connectionDot,
                  isConnected ? cameraStyles.connectionDotConnected : cameraStyles.connectionDotOffline,
                ]}
              />
              <Text style={cameraStyles.connectionText}>{connectionLabel}</Text>
            </View>
          <TouchableOpacity 
            style={cameraStyles.cameraFlipButton} 
            onPress={() => {
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
              toggleCameraFacing();
            }}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel={`Switch to ${isFrontCamera ? 'back' : 'front'} camera`}
            accessibilityHint="Activates to flip camera view"
            hitSlop={{ top: 15, bottom: 15, left: 15, right: 15 }}
          >
            <Image
              source={require('../../images/camera.png')}
              style={cameraStyles.cameraFlipIcon}
              resizeMode="contain"
              accessibilityIgnoresInvertColors={true}
            />
          </TouchableOpacity>
          </View>
        </View>

        <Camera
          ref={camera}
          style={cameraStyles.camera}
          device={device}
          isActive={true}
          video={true}
          photo={true}
          audio={false}
          onError={onError}
        />

        <View pointerEvents="none" style={cameraStyles.previewOverlay}>
          <View style={cameraStyles.handGuide}>
            <View style={[cameraStyles.guideCorner, cameraStyles.guideCornerTopLeft]} />
            <View style={[cameraStyles.guideCorner, cameraStyles.guideCornerTopRight]} />
            <View style={[cameraStyles.guideCorner, cameraStyles.guideCornerBottomLeft]} />
            <View style={[cameraStyles.guideCorner, cameraStyles.guideCornerBottomRight]} />
            {!isActive && (
              <View style={cameraStyles.guideHint}>
                <Text style={cameraStyles.guideHintText}>Place your hands inside the frame</Text>
              </View>
            )}
          </View>
        </View>

        <View 
          style={cameraStyles.translationContainer}
          accessible={true}
          accessibilityRole="text"
          accessibilityLabel="Translation result"
        >
          <View style={cameraStyles.translationHeader}>
            <View>
              <Text style={cameraStyles.translationStatus}>{sessionStatus}</Text>
              <Text style={cameraStyles.translationHint}>{sessionHint}</Text>
            </View>
            {(captureStatus || apiError || wsError || isSending) && (
              <Text
                style={[
                  cameraStyles.captureStatusText,
                  (apiError || wsError) ? cameraStyles.captureStatusError : null,
                ]}
              >
                {apiError || wsError || (isSending ? 'Uploading...' : captureStatus)}
              </Text>
            )}
          </View>

          <View 
            style={cameraStyles.translationContent}
            accessible={true}
            accessibilityLiveRegion="assertive"
            accessibilityLabel={translation || "Waiting for translation"}
          >
            {translation ? (
              <Text 
                style={cameraStyles.translationText}
                accessible={true}
                accessibilityRole="text"
              >
                {translation}
              </Text>
            ) : isTranslating ? (
              <Text 
                style={cameraStyles.placeholderText}
                accessible={true}
                accessibilityRole="text"
              >
                Translating...
              </Text>
            ) : (
              <Text 
                style={cameraStyles.placeholderText}
                accessible={true}
                accessibilityRole="text"
              >
                Your translation will appear here
              </Text>
            )}
            {isActive && lastStatus === 'idle' && !translation && (
              <Text 
                style={{
                  color: '#ffd700',
                  fontSize: 12,
                  textAlign: 'center',
                  marginTop: 8,
                }}
                accessible={true}
                accessibilityRole="text"
              >
                No hands detected
              </Text>
            )}
          </View>
          
          <TouchableOpacity 
            style={[cameraStyles.audioButton, isPlaying && cameraStyles.audioButtonActive]} 
            onPress={handleTextToSpeech}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel={isPlaying ? "Stop speech" : "Speak translation"}
            accessibilityHint={isPlaying ? "Stops the current speech" : "Reads the translation aloud"}
            accessibilityState={{ 
              selected: isPlaying,
              busy: isPlaying 
            }}
            hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
          >
            <Image
              source={require('../../images/volume.png')}
              style={[cameraStyles.audioButtonIcon, isPlaying ? cameraStyles.audioButtonIconActive : null]}
              resizeMode="contain"
              accessibilityIgnoresInvertColors={true}
            />
          </TouchableOpacity>
        </View>

        <View style={cameraStyles.bottomActionsContainer}>
          <TouchableOpacity 
            style={cameraStyles.actionButton} 
            onPress={handleTranslatePress}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Translation options"
            accessibilityHint="Opens translation settings and options"
            hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
          >
            <Image
              source={require('../../images/translate.png')}
              style={cameraStyles.actionButtonImage}
              resizeMode="contain"
              accessibilityIgnoresInvertColors={true}
            />
          </TouchableOpacity>

          <TouchableOpacity
            style={[
              cameraStyles.tapToStartButton,
              isActive ? cameraStyles.tapToStartButtonActive : null,
              isFinishingSession ? cameraStyles.tapToStartButtonBusy : null,
            ]}
            disabled={isFinishingSession}
            onPress={async () => {
              await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
              handleTapToStart();
            }}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel={actionAccessibilityLabel}
            accessibilityHint={
              isActive
                ? "Stops capturing sign language and processes remaining frames"
                : "Starts capturing sign language from camera"
            }
            accessibilityState={{
              disabled: isFinishingSession,
              busy: isFinishingSession
            }}
            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
          >
            <Image
              source={require('../../images/taptostart.png')}
              style={[
                cameraStyles.tapToStartIcon,
                isActive ? cameraStyles.tapToStartIconActive : null,
              ]}
              resizeMode="contain"
              accessibilityIgnoresInvertColors={true}
            />
            <Text style={cameraStyles.tapToStartText}>
              {actionLabel}
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
}
