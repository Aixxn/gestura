import { StyleSheet } from 'react-native';

// --- Color Palette ---
export const COLORS = {
  primaryBlue: '#347ae2ff', 
  accentBlue: '#0917b4ff', 
  background: '#535e7eff', 
  textPrimary: '#ffffffff', 
  textSecondary: '#929bf4ff',
  white: '#FFFFFF',
  lightBlueCircle: 'rgba(0, 122, 255, 0.15)', 
  dotInactive: '#D1D5DB',
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
  topBar: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 10,
    height: 50,
    position: 'relative',
  },
  logoContainer: {
    flex: 1,
    alignItems: 'center',
  },
  logoIcon: {
    fontSize: 28,
  },
  closeButton: {
    position: 'absolute',
    right: 20,
    top: 10,
    padding: 5,
  },
  closeIcon: {
    fontSize: 32,
    color: COLORS.textSecondary,
    fontWeight: '300',
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
  gradientBackground: {
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 12,
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
    marginTop: 100,
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