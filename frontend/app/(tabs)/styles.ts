import { StyleSheet } from 'react-native';

// --- Color Palette ---
export const COLORS = {
  primaryBlue: '#347ae2ff', 
  accentBlue: '#0917b4ff', 
  background: '#535e7eff', 
  textPrimary: '#ffffffff', 
  textSecondary: '#929bf4ff',
  buttonGrey: '#c3c2c2d9',
  buttonBorder: '#c3c2c2d9',
  white: '#FFFFFF',
  lightBlueCircle: 'rgba(0, 122, 255, 0.15)', 
  dotInactive: '#D1D5DB',
  tapButtonGrey: '#F5F5F5',
};

export const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  backgroundImage: {
    flex: 1,
    width: '100%',
    height: '100%',
  },
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(42, 82, 192, 0.14)',
  },
  
  content: {
    alignItems: 'center',
    paddingHorizontal: 30,
    paddingVertical: 20,
  },
  titleContainer: {
    marginTop: 130,
    marginBottom: 0,
  },
  
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: COLORS.white,
    textAlign: 'center',
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
    marginTop: 140,
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
    backgroundColor: COLORS.primaryBlue,
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

export const cameraStyles = StyleSheet.create({
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
    backgroundColor: COLORS.accentBlue,
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
    height: 700,
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
    bottom: 35,
    left: 0,
    right: 0,
    height: 250,
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
    backgroundColor: COLORS.accentBlue,
  },
  bottomActionsContainer: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 60,
    gap: 20, // Adds consistent spacing between buttons
  },
  actionButton: {
    backgroundColor: COLORS.tapButtonGrey,
    width: 55,
    height: 55,
    borderRadius: 27.5,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  actionButtonImage: {
    width: 24,
    height: 24,
    tintColor: '#333333',
  },
  tapToStartButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.tapButtonGrey,
    borderRadius: 30,
    paddingVertical: 15,
    paddingHorizontal: 25,
    minWidth: 140, // Slightly reduced width for better balance
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  tapToStartIcon: {
    width: 20,
    height: 20,
    marginRight: 8,
    tintColor: '#333333',
  },
  tapToStartText: {
    fontSize: 16,
    color: '#333333',
    fontWeight: '600',
  },
});