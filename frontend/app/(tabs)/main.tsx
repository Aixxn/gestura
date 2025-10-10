/* import { runOnJS } from 'react-native-reanimated';
import { useFrameProcessor } from 'react-native-vision-camera';

// API configuration
const API_BASE_URL = 'https://your-api-endpoint.com'; // Replace with your actual API URL

// Function to send frame to your OpenCV API
const processFrameWithAPI = async (frameData: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}/process-frame`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        frameData: frameData,
        // Add any additional parameters your API needs
        width: 640,
        height: 480,
        processingOptions: {
          detectGestures: true,
          enhanceImage: true,
        }
      }),
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error processing frame with API:', error);
    return null;
  }
};

const frameProcessor = useFrameProcessor((frame) => {
  'worklet';

  // Convert frame to base64 for API transmission
  try {
    // Convert frame to base64 string (you may need to adjust this based on your frame format)
    const base64Frame = frame.toString(); // Remove the 'base64' parameter
    
    // Create a function to handle the API call
    const handleAPICall = (frameData: string) => {
      processFrameWithAPI(frameData).then((result: any) => {
        if (result) {
          // Handle the API response
          console.log('Gesture detected:', result.gestures);
          console.log('Processed image:', result.processedImage);
          
          // You can emit events, update state, etc. based on the result
          // Example: EventEmitter.emit('gestureDetected', result.gestures);
        }
      }).catch((error) => {
        console.error('API call failed:', error);
      });
    };
    
    // Send to API using runOnJS to bridge from worklet to JS context
    runOnJS(handleAPICall)(base64Frame);
    
  } catch (error) {
    console.error('Error in frame processor:', error);
  }

}, []);

// Make sure to export or use the frameProcessor
export { frameProcessor };
*/
