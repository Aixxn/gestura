import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import {
  Alert,
  Linking,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

import { useTranslationHistory } from '../../hooks/useTranslationHistory';
import { logoutUser } from '../../services/auth';
import { clearTranslationHistory, type TranslationHistoryEntry } from '../../services/translationHistory';

const toVideoUri = (path: string) => (path.startsWith('file://') ? path : `file://${path}`);

const formatDate = (iso: string) => (
  new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(iso))
);

const HistoryVideo = ({ entry }: { entry: TranslationHistoryEntry }) => {
  const openVideo = async () => {
    const source = toVideoUri(entry.videoPath);

    try {
      const canOpen = await Linking.canOpenURL(source);
      if (!canOpen) {
        Alert.alert('Video unavailable', 'This recorded video cannot be opened on this device.');
        return;
      }

      await Linking.openURL(source);
    } catch {
      Alert.alert('Video unavailable', 'This recorded video cannot be opened on this device.');
    }
  };

  return (
    <TouchableOpacity
      style={styles.video}
      onPress={openVideo}
      accessibilityRole="button"
      accessibilityLabel="Open recorded translation video"
    >
      <Ionicons name="play-circle" size={42} color="#fff" />
      <Text style={styles.videoText}>Open recorded video</Text>
    </TouchableOpacity>
  );
};

export default function SettingsScreen() {
  const router = useRouter();
  const history = useTranslationHistory();
  const [isLoggingOut, setIsLoggingOut] = useState(false);

  const handleLogout = async () => {
    if (isLoggingOut) return;

    setIsLoggingOut(true);
    try {
      await logoutUser();
      router.replace('/(tabs)/login');
    } catch {
      router.replace('/(tabs)/login');
    } finally {
      setIsLoggingOut(false);
    }
  };

  const handleClearHistory = () => {
    Alert.alert(
      'Clear history',
      'Remove all translation history from this app session?',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Clear', style: 'destructive', onPress: clearTranslationHistory },
      ]
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.iconButton}
          onPress={() => router.back()}
          accessibilityRole="button"
          accessibilityLabel="Go back"
        >
          <Ionicons name="chevron-back" size={24} color="#10233f" />
        </TouchableOpacity>
        <Text style={styles.title}>Profile</Text>
        <View style={styles.iconButton} />
      </View>

      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.profileBand}>
          <View style={styles.avatar}>
            <Ionicons name="person" size={30} color="#fff" />
          </View>
          <View style={styles.profileCopy}>
            <Text style={styles.profileName}>Gestura User</Text>
            <Text style={styles.profileMeta}>Translation history and account settings</Text>
          </View>
        </View>

        <View style={styles.actionsRow}>
          <TouchableOpacity
            style={styles.secondaryButton}
            onPress={handleClearHistory}
            accessibilityRole="button"
            accessibilityLabel="Clear translation history"
          >
            <Ionicons name="trash-outline" size={18} color="#17345f" />
            <Text style={styles.secondaryButtonText}>Clear History</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.logoutButton}
            onPress={handleLogout}
            accessibilityRole="button"
            accessibilityLabel="Logout"
            disabled={isLoggingOut}
          >
            <Ionicons name="log-out-outline" size={18} color="#fff" />
            <Text style={styles.logoutButtonText}>
              {isLoggingOut ? 'Logging out...' : 'Logout'}
            </Text>
          </TouchableOpacity>
        </View>

        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Translation History</Text>
          <Text style={styles.sectionCount}>{history.length}</Text>
        </View>

        {history.length === 0 ? (
          <View style={styles.emptyState}>
            <Ionicons name="videocam-outline" size={32} color="#64748b" />
            <Text style={styles.emptyTitle}>No translations yet</Text>
            <Text style={styles.emptyBody}>
              Completed translations will appear here with their recorded video.
            </Text>
          </View>
        ) : (
          history.map((entry) => (
            <View key={entry.id} style={styles.historyItem}>
              <HistoryVideo entry={entry} />
              <View style={styles.historyContent}>
                <Text style={styles.historyDate}>{formatDate(entry.createdAt)}</Text>
                <Text style={styles.historyLabel}>ASL Gloss</Text>
                <Text style={styles.historyGloss}>{entry.aslGloss || 'No gloss detected'}</Text>
                <Text style={styles.historyLabel}>English</Text>
                <Text style={styles.historyEnglish}>{entry.english}</Text>
              </View>
            </View>
          ))
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#eef3fb',
  },
  header: {
    minHeight: 60,
    paddingHorizontal: 18,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  iconButton: {
    width: 42,
    height: 42,
    borderRadius: 21,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 20,
    fontWeight: '800',
    color: '#10233f',
  },
  content: {
    paddingHorizontal: 18,
    paddingBottom: 32,
    gap: 16,
  },
  profileBand: {
    minHeight: 96,
    borderRadius: 8,
    backgroundColor: '#10233f',
    padding: 18,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
  },
  avatar: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#2563eb',
    alignItems: 'center',
    justifyContent: 'center',
  },
  profileCopy: {
    flex: 1,
  },
  profileName: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '800',
  },
  profileMeta: {
    color: 'rgba(255,255,255,0.74)',
    fontSize: 12,
    marginTop: 4,
    fontWeight: '600',
  },
  actionsRow: {
    flexDirection: 'row',
    gap: 10,
  },
  secondaryButton: {
    flex: 1,
    minHeight: 48,
    borderRadius: 8,
    backgroundColor: '#dbe8ff',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  secondaryButtonText: {
    color: '#17345f',
    fontWeight: '800',
  },
  logoutButton: {
    flex: 1,
    minHeight: 48,
    borderRadius: 8,
    backgroundColor: '#dc2626',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  logoutButtonText: {
    color: '#fff',
    fontWeight: '800',
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 6,
  },
  sectionTitle: {
    color: '#10233f',
    fontSize: 18,
    fontWeight: '800',
  },
  sectionCount: {
    minWidth: 30,
    textAlign: 'center',
    color: '#2563eb',
    fontWeight: '800',
  },
  emptyState: {
    minHeight: 180,
    borderRadius: 8,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 22,
  },
  emptyTitle: {
    marginTop: 10,
    color: '#10233f',
    fontSize: 16,
    fontWeight: '800',
  },
  emptyBody: {
    marginTop: 6,
    color: '#64748b',
    fontSize: 13,
    lineHeight: 19,
    textAlign: 'center',
    fontWeight: '600',
  },
  historyItem: {
    borderRadius: 8,
    backgroundColor: '#fff',
    overflow: 'hidden',
  },
  video: {
    width: '100%',
    aspectRatio: 16 / 9,
    backgroundColor: '#0f172a',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  videoText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '800',
  },
  historyContent: {
    padding: 14,
  },
  historyDate: {
    color: '#64748b',
    fontSize: 12,
    fontWeight: '700',
    marginBottom: 10,
  },
  historyLabel: {
    color: '#2563eb',
    fontSize: 11,
    fontWeight: '800',
    textTransform: 'uppercase',
    marginTop: 8,
  },
  historyGloss: {
    color: '#10233f',
    fontSize: 18,
    lineHeight: 24,
    fontWeight: '800',
    marginTop: 3,
  },
  historyEnglish: {
    color: '#17345f',
    fontSize: 16,
    lineHeight: 22,
    fontWeight: '700',
    marginTop: 3,
  },
});
