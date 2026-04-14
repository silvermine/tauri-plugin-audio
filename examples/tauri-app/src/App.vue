<template>
   <div class="app">
      <h1>Audio Player Demo</h1>

      <!-- Load bar -->
      <div class="load-bar">
         <input
            v-model="sourceUrl"
            type="text"
            placeholder="File path or URL"
            @keydown.enter="loadTrack"
         />
         <button class="load-btn" @click="loadTrack" :disabled="!canLoad || isLoading">
            Load
         </button>
      </div>

      <div class="player">
         <!-- Artwork (left) -->
         <div class="player-artwork">
            <img
               v-if="player?.artwork"
               :src="player.artwork"
               class="artwork"
               alt="Album artwork"
            />
            <div v-else class="artwork placeholder">
               <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55C7.79 13 6 14.79 6 17s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/>
               </svg>
            </div>
         </div>

         <!-- Content (right) -->
         <div class="player-content">
            <!-- Progress bar -->
            <input
               type="range"
               class="progress-bar"
               :class="{ inactive: !canSeek }"
               :value="currentTime"
               :max="duration || 1"
               step="0.1"
               @input="canSeek && seekTo(Number(($event.target as HTMLInputElement).value))"
            />

            <!-- Title + time -->
            <div class="track-row">
               <span class="title">{{ trackTitle }}</span>
               <span class="time">{{ timeDisplay }}</span>
            </div>

            <!-- Controls row -->
            <div class="controls-row">
               <!-- Play / Pause -->
               <button class="ctrl-btn large" @click="togglePlay" :disabled="!canPlay && !canPause">
                  <svg v-if="isPlaying" viewBox="0 0 24 24" fill="currentColor">
                     <rect x="6" y="4" width="4" height="16" rx="1"/>
                     <rect x="14" y="4" width="4" height="16" rx="1"/>
                  </svg>
                  <svg v-else viewBox="0 0 24 24" fill="currentColor">
                     <path d="M8 5v14l11-7z"/>
                  </svg>
               </button>

               <!-- Seek -5s -->
               <button class="ctrl-btn seek" @click="seekRelative(-5)" :disabled="!canSeek">
                  <svg viewBox="0 0 24 24" fill="currentColor">
                     <path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/>
                  </svg>
                  <span class="seek-label">5</span>
               </button>

               <!-- Seek +15s -->
               <button class="ctrl-btn seek" @click="seekRelative(15)" :disabled="!canSeek">
                  <svg viewBox="0 0 24 24" fill="currentColor">
                     <path d="M12 5V1l5 5-5 5V7c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6h2c0 4.42-3.58 8-8 8s-8-3.58-8-8 3.58-8 8-8z"/>
                  </svg>
                  <span class="seek-label">15</span>
               </button>

               <!-- Volume -->
               <button class="ctrl-btn" @click="toggleMute">
                  <svg v-if="player?.muted || (player?.volume ?? 1) === 0" viewBox="0 0 24 24" fill="currentColor">
                     <path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51A8.796 8.796 0 0021 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/>
                  </svg>
                  <svg v-else viewBox="0 0 24 24" fill="currentColor">
                     <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
                  </svg>
               </button>

               <div class="volume-control">
                  <input
                     type="range"
                     class="volume-slider"
                     :value="player?.volume ?? 1"
                     min="0"
                     max="1"
                     step="0.01"
                     @input="setVolume(Number(($event.target as HTMLInputElement).value))"
                  />
                  <span class="volume-tooltip">{{ volumePercent }}</span>
               </div>

               <!-- Settings toggle -->
               <div class="settings-anchor" ref="settingsAnchorEl">
                  <button class="ctrl-btn large" @click="toggleSettingsMenu">
                     <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58a.49.49 0 00.12-.61l-1.92-3.32a.488.488 0 00-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.484.484 0 00-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.07.62-.07.94s.02.64.07.94l-2.03 1.58a.49.49 0 00-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>
                     </svg>
                  </button>

                  <!-- Settings popover menu -->
                  <div v-if="showSettings" class="settings-menu" @click.stop>
                     <!-- Main menu -->
                     <template v-if="settingsView === 'main'">
                        <button class="menu-item" @click="settingsView = 'speed'">
                           <svg class="menu-icon" viewBox="0 0 24 24" fill="currentColor">
                              <path d="M10 8v8l6-4-6-4zm1.5 2.83L13.54 12l-2.04 1.17V10.83zM20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 14H4V6h16v12z"/>
                           </svg>
                           <span class="menu-label">Speed</span>
                           <span class="menu-value">{{ speedLabel }}</span>
                           <svg class="menu-chevron" viewBox="0 0 24 24" fill="currentColor">
                              <path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/>
                           </svg>
                        </button>
                        <button class="menu-item" @click="toggleLoop">
                           <svg class="menu-icon" viewBox="0 0 24 24" fill="currentColor">
                              <path d="M7 7h10v3l4-4-4-4v3H5v6h2V7zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2v4z"/>
                           </svg>
                           <span class="menu-label">Loop</span>
                           <span class="menu-value">{{ player?.loop ? 'On' : 'Off' }}</span>
                        </button>
                        <button v-if="canStop" class="menu-item destructive" @click="stopTrack">
                           <svg class="menu-icon" viewBox="0 0 24 24" fill="currentColor">
                              <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
                           </svg>
                           <span class="menu-label">Unload</span>
                        </button>
                     </template>

                     <!-- Speed submenu -->
                     <template v-if="settingsView === 'speed'">
                        <button class="menu-item menu-back" @click="settingsView = 'main'">
                           <svg class="menu-icon" viewBox="0 0 24 24" fill="currentColor">
                              <path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z"/>
                           </svg>
                           <span class="menu-label">Speed</span>
                        </button>
                        <div class="menu-divider"></div>
                        <button
                           v-for="opt in speedOptions"
                           :key="opt.value"
                           class="menu-item"
                           :class="{ active: (player?.playbackRate ?? 1) === opt.value }"
                           @click="selectSpeed(opt.value)"
                        >
                           <svg v-if="(player?.playbackRate ?? 1) === opt.value" class="menu-icon check" viewBox="0 0 24 24" fill="currentColor">
                              <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
                           </svg>
                           <span v-else class="menu-icon"></span>
                           <span class="menu-label">{{ opt.label }}</span>
                        </button>
                     </template>
                  </div>
               </div>
            </div>
         </div>

         <!-- Error display -->
         <div v-if="errorMessage" class="error">{{ errorMessage }}</div>
      </div>

      <!-- Event log -->
      <div class="event-log">
         <div class="event-log-header">
            <h2>Events</h2>
            <div class="event-log-actions">
               <label class="event-filter">
                  <input type="checkbox" v-model="showTimeUpdates" />
                  Time updates
               </label>
               <button class="event-clear-btn" @click="eventLog = []; lastStateSnapshot = null">Clear</button>
            </div>
         </div>
         <div class="event-log-entries" ref="eventLogEl">
            <template v-for="(entry, i) in filteredEvents" :key="i">
               <div class="event-entry" :class="entry.type">
                  <div class="event-row" @click="entry.expanded = !entry.expanded">
                     <span class="event-badge" :class="entry.type">{{ entry.label }}</span>
                     <span class="event-summary">{{ entry.summary }}</span>
                     <span class="event-time">{{ entry.timestamp }}</span>
                  </div>
                  <pre v-if="entry.expanded" class="event-payload">{{ entry.payload }}</pre>
               </div>
            </template>
            <div v-if="filteredEvents.length === 0" class="event-empty">No events yet</div>
         </div>
      </div>
   </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, onMounted, onUnmounted } from 'vue';
import './style.css';
import {
   getPlayer,
   PlaybackStatus,
   hasAction,
   AudioAction,
} from '@silvermine/tauri-plugin-audio';
import type { PlayerWithAnyStatus, AudioMetadata } from '@silvermine/tauri-plugin-audio';

const player = ref<PlayerWithAnyStatus | null>(null);
const currentTime = ref(0);
const duration = ref(0);
const SAMPLE_TRACK = {
   src: 'https://www.learningcontainer.com/wp-content/uploads/2020/02/Kalimba.mp3',
   title: 'Kalimba',
   artwork: 'https://cdn.pixabay.com/photo/2024/02/28/07/42/european-shorthair-8601492_1280.jpg',
};

const sourceUrl = ref(SAMPLE_TRACK.src);
const showSettings = ref(false);
const settingsView = ref<'main' | 'speed'>('main');
const settingsAnchorEl = ref<HTMLElement | null>(null);
const isLoading = ref(false);
const errorMessage = ref('');

const speedOptions = [
   { label: '0.25x', value: 0.25 },
   { label: '0.5x', value: 0.5 },
   { label: '0.75x', value: 0.75 },
   { label: '1.0x', value: 1 },
   { label: '1.25x', value: 1.25 },
   { label: '1.5x', value: 1.5 },
   { label: '1.75x', value: 1.75 },
   { label: '2x', value: 2 },
];

let unlistenState: (() => void) | null = null;
let unlistenTime: (() => void) | null = null;

const MAX_LOG_ENTRIES = 200;

interface EventLogEntry {
   type: 'state' | 'time';
   label: string;
   summary: string;
   timestamp: string;
   payload: string;
   expanded: boolean;
}

const eventLog = ref<EventLogEntry[]>([]);
const showTimeUpdates = ref(false);
const eventLogEl = ref<HTMLElement | null>(null);
let lastStateSnapshot: Record<string, unknown> | null = null;

const filteredEvents = computed(() => {
   if (showTimeUpdates.value) {
      return eventLog.value;
   }
   return eventLog.value.filter((e) => e.type === 'state');
});

function formatTimestamp(): string {
   const now = new Date(),
         h = String(now.getHours()).padStart(2, '0'),
         m = String(now.getMinutes()).padStart(2, '0'),
         s = String(now.getSeconds()).padStart(2, '0'),
         ms = String(now.getMilliseconds()).padStart(3, '0');

   return `${h}:${m}:${s}.${ms}`;
}

function pushEvent(entry: EventLogEntry): void {
   const log = eventLog.value;

   log.push(entry);
   if (log.length > MAX_LOG_ENTRIES) {
      log.splice(0, log.length - MAX_LOG_ENTRIES);
   }
   nextTick(() => {
      if (eventLogEl.value) {
         eventLogEl.value.scrollTop = eventLogEl.value.scrollHeight;
      }
   });
}

function summarizeStateChange(updated: PlayerWithAnyStatus): string {
   const curr: Record<string, unknown> = { ...updated };
   const prev = lastStateSnapshot;
   const parts: string[] = [];

   // Status transition
   if (!prev || prev.status !== curr.status) {
      parts.push(prev ? `${prev.status} → ${curr.status}` : `${curr.status}`);
   }

   // Track change
   if (!prev || prev.src !== curr.src) {
      const name = updated.title || (updated.src ? updated.src.split('/').pop() : null);

      if (name) {
         parts.push(name);
      }
   }

   // Setting changes
   if (prev) {
      if (prev.volume !== curr.volume) {
         parts.push(`volume: ${Math.round((curr.volume as number) * 100)}%`);
      }
      if (prev.muted !== curr.muted) {
         parts.push(curr.muted ? 'muted' : 'unmuted');
      }
      if (prev.playbackRate !== curr.playbackRate) {
         parts.push(`rate: ${curr.playbackRate}x`);
      }
      if (prev.loop !== curr.loop) {
         parts.push(curr.loop ? 'loop: on' : 'loop: off');
      }
      if (prev.duration !== curr.duration && (curr.duration as number) > 0) {
         parts.push(`duration: ${formatTime(curr.duration as number)}`);
      }
      if (prev.currentTime !== curr.currentTime && prev.status === curr.status) {
         parts.push(`seek: ${formatTime(curr.currentTime as number)}`);
      }
   }

   lastStateSnapshot = curr;

   if (parts.length === 0) {
      return `${updated.status} · ${formatTime(updated.currentTime)}`;
   }

   return parts.join(' · ');
}

// -- Computed --

const trackTitle = computed(() => {
   if (!player.value || player.value.status === PlaybackStatus.Idle) {
      return 'No track loaded';
   }
   return player.value.title || player.value.src?.split('/').pop() || 'Unknown';
});

const timeDisplay = computed(() => {
   return `${formatTime(currentTime.value)} / ${formatTime(duration.value)}`;
});

const isPlaying = computed(() => player.value?.status === PlaybackStatus.Playing);

const canLoad = computed(() => {
   return player.value
      && (hasAction(player.value, AudioAction.Load) || hasAction(player.value, AudioAction.Stop))
      && sourceUrl.value.length > 0;
});

const canPlay = computed(() => {
   return player.value && hasAction(player.value, AudioAction.Play);
});

const canPause = computed(() => {
   return player.value && hasAction(player.value, AudioAction.Pause);
});

const canSeek = computed(() => {
   return player.value && hasAction(player.value, AudioAction.Seek);
});

const canStop = computed(() => {
   return player.value && hasAction(player.value, AudioAction.Stop);
});

const volumePercent = computed(() => {
   return `${Math.round((player.value?.volume ?? 1) * 100)}`;
});

const speedLabel = computed(() => {
   const rate = player.value?.playbackRate ?? 1;
   const opt = speedOptions.find((o) => o.value === rate);

   return opt ? opt.label : `${rate}x`;
});

// -- Helpers --

function formatTime(seconds: number): string {
   const m = Math.floor(seconds / 60);
   const s = Math.floor(seconds % 60);

   return `${m}:${s.toString().padStart(2, '0')}`;
}

function toggleSettingsMenu(): void {
   showSettings.value = !showSettings.value;
   settingsView.value = 'main';
}

function onClickOutside(e: MouseEvent): void {
   if (settingsAnchorEl.value && !settingsAnchorEl.value.contains(e.target as Node)) {
      showSettings.value = false;
      settingsView.value = 'main';
   }
}

async function selectSpeed(rate: number): Promise<void> {
   await setPlaybackRate(rate);
   showSettings.value = false;
   settingsView.value = 'main';
}

// -- Actions --

async function loadTrack(): Promise<void> {
   if (!player.value || !sourceUrl.value) {
      return;
   }

   isLoading.value = true;
   errorMessage.value = '';

   try {
      // Stop current track first if needed
      if (hasAction(player.value, AudioAction.Stop)) {
         const stopResp = await player.value.stop();

         player.value = stopResp.player;
      }

      if (!hasAction(player.value, AudioAction.Load)) {
         return;
      }

      const isSample = sourceUrl.value === SAMPLE_TRACK.src;
      const metadata: AudioMetadata = {
         title: isSample ? SAMPLE_TRACK.title : sourceUrl.value.split('/').pop() || 'Unknown Track',
         ...(isSample && { artwork: SAMPLE_TRACK.artwork }),
      };
      const resp = await player.value.load(sourceUrl.value, metadata);

      player.value = resp.player;
      duration.value = resp.player.duration;
      currentTime.value = 0;
   } catch (e) {
      errorMessage.value = `Failed to load: ${e}`;
   } finally {
      isLoading.value = false;
   }
}

async function togglePlay(): Promise<void> {
   if (!player.value) {
      return;
   }

   errorMessage.value = '';

   try {
      if (hasAction(player.value, AudioAction.Play)) {
         const resp = await player.value.play();

         player.value = resp.player;
      } else if (hasAction(player.value, AudioAction.Pause)) {
         const resp = await player.value.pause();

         player.value = resp.player;
      }
   } catch (e) {
      errorMessage.value = `${e}`;
   }
}

async function seekRelative(offset: number): Promise<void> {
   if (!player.value || !hasAction(player.value, AudioAction.Seek)) {
      return;
   }

   const pos = Math.max(0, currentTime.value + offset);
   const resp = await player.value.seek(pos);

   player.value = resp.player;
   currentTime.value = resp.player.currentTime;
}

async function seekTo(position: number): Promise<void> {
   if (!player.value || !hasAction(player.value, AudioAction.Seek)) {
      return;
   }

   const resp = await player.value.seek(position);

   player.value = resp.player;
   currentTime.value = resp.player.currentTime;
}

async function setVolume(level: number): Promise<void> {
   if (!player.value) {
      return;
   }
   player.value = await player.value.setVolume(level);
}

async function toggleMute(): Promise<void> {
   if (!player.value) {
      return;
   }
   player.value = await player.value.setMuted(!player.value.muted);
}

async function setPlaybackRate(rate: number): Promise<void> {
   if (!player.value) {
      return;
   }
   player.value = await player.value.setPlaybackRate(rate);
}

async function toggleLoop(): Promise<void> {
   if (!player.value) {
      return;
   }
   player.value = await player.value.setLoop(!player.value.loop);
}

async function stopTrack(): Promise<void> {
   if (!player.value || !hasAction(player.value, AudioAction.Stop)) {
      return;
   }

   const resp = await player.value.stop();

   player.value = resp.player;
   currentTime.value = 0;
   duration.value = 0;
   showSettings.value = false;
}

// -- Lifecycle --

onMounted(async () => {
   document.addEventListener('click', onClickOutside);

   const p = await getPlayer();

   player.value = p;

   unlistenState = await p.listen((updated) => {
      player.value = updated;
      currentTime.value = updated.currentTime;
      duration.value = updated.duration;
      pushEvent({
         type: 'state',
         label: updated.status,
         summary: summarizeStateChange(updated),
         timestamp: formatTimestamp(),
         payload: JSON.stringify(updated, null, 2),
         expanded: false,
      });
   });

   unlistenTime = await p.onTimeUpdate((time) => {
      currentTime.value = time.currentTime;
      if (time.duration > 0) {
         duration.value = time.duration;
      }
      pushEvent({
         type: 'time',
         label: 'time',
         summary: `${formatTime(time.currentTime)} / ${formatTime(time.duration)}`,
         timestamp: formatTimestamp(),
         payload: JSON.stringify(time, null, 2),
         expanded: false,
      });
   });

   // Preload sample track
   if (hasAction(p, AudioAction.Load)) {
      try {
         isLoading.value = true;
         const resp = await p.load(SAMPLE_TRACK.src, {
            title: SAMPLE_TRACK.title,
            artwork: SAMPLE_TRACK.artwork,
         });

         player.value = resp.player;
         duration.value = resp.player.duration;
      } catch (e) {
         errorMessage.value = `Failed to preload: ${e}`;
      } finally {
         isLoading.value = false;
      }
   }
});

onUnmounted(() => {
   document.removeEventListener('click', onClickOutside);
   unlistenState?.();
   unlistenTime?.();
});
</script>

<style scoped>
.app {
   max-width: 700px;
   margin: 0 auto;
   padding: 24px 16px;
}

h1 {
   font-size: 20px;
   font-weight: 700;
   margin-bottom: 16px;
}

.player {
   background: #2c2c2e;
   border-radius: 10px;
   padding: 12px;
   display: flex;
   gap: 12px;
}

.player-artwork {
   flex-shrink: 0;
   align-self: center;
   width: 80px;
   height: 80px;
}

.player-content {
   flex: 1;
   min-width: 0;
   display: flex;
   flex-direction: column;
   gap: 6px;
   justify-content: center;
}

/* Load bar */
.load-bar {
   display: flex;
   gap: 8px;
   margin-bottom: 12px;
}

.load-bar input[type="text"] {
   flex: 1;
}

.load-btn {
   font-size: 13px;
   font-weight: 600;
   background: #0a84ff;
   color: #fff;
   border: none;
   border-radius: 6px;
   padding: 8px 16px;
   cursor: pointer;
   white-space: nowrap;
}

.load-btn:hover:not(:disabled) {
   background: #409cff;
}

.load-btn:disabled {
   opacity: 0.4;
   cursor: not-allowed;
}

/* Progress bar */
.progress-bar {
   width: 100%;
   height: 12px;
   margin: 0;
}

.progress-bar.inactive {
   pointer-events: none;
}

.progress-bar.inactive::-webkit-slider-thumb {
   visibility: hidden;
}

/* Track info */
.track-row {
   display: flex;
   align-items: center;
   gap: 12px;
}

.artwork {
   width: 80px;
   height: 80px;
   border-radius: 6px;
   object-fit: cover;
}

.artwork.placeholder {
   background: #48484a;
   display: flex;
   align-items: center;
   justify-content: center;
   color: #98989d;
}

.artwork.placeholder svg {
   width: 28px;
   height: 28px;
}

.title {
   font-size: 15px;
   font-weight: 600;
   flex: 1;
   overflow: hidden;
   text-overflow: ellipsis;
   white-space: nowrap;
}

.time {
   font-size: 13px;
   color: #98989d;
   font-variant-numeric: tabular-nums;
   white-space: nowrap;
}

/* Controls */
.controls-row {
   display: flex;
   align-items: center;
   gap: 8px;
}

.ctrl-btn {
   width: 28px;
   height: 28px;
   display: flex;
   align-items: center;
   justify-content: center;
   flex-shrink: 0;
   color: #f5f5f7;
   border-radius: 4px;
}

.ctrl-btn:hover:not(:disabled) {
   background: rgba(255, 255, 255, 0.08);
}

.ctrl-btn svg {
   width: 20px;
   height: 20px;
}

.ctrl-btn.large {
   width: 32px;
   height: 32px;
}

.ctrl-btn.large svg {
   width: 24px;
   height: 24px;
}

.ctrl-btn.seek {
   position: relative;
   width: 32px;
   height: 32px;
}

.ctrl-btn.seek svg {
   width: 28px;
   height: 28px;
}

.seek-label {
   position: absolute;
   font-size: 8px;
   font-weight: 700;
   pointer-events: none;
   top: 50%;
   left: 50%;
   transform: translate(-50%, -30%);
}

.volume-control {
   position: relative;
   display: flex;
   align-items: center;
}

.volume-slider {
   width: 80px;
   flex-shrink: 0;
}

.volume-tooltip {
   position: absolute;
   bottom: calc(100% + 6px);
   left: 50%;
   transform: translateX(-50%);
   font-size: 11px;
   font-weight: 600;
   color: #f5f5f7;
   background: #48484a;
   padding: 2px 6px;
   border-radius: 4px;
   white-space: nowrap;
   pointer-events: none;
   opacity: 0;
   transition: opacity 0.15s;
}

.volume-control:hover .volume-tooltip {
   opacity: 1;
}

/* Settings menu */
.settings-anchor {
   position: relative;
   margin-left: auto;
}

.settings-menu {
   position: absolute;
   bottom: calc(100% + 8px);
   right: 0;
   min-width: 200px;
   max-height: 180px;
   overflow-y: auto;
   background: #2c2c2e;
   border: 1px solid #48484a;
   border-radius: 8px;
   padding: 4px;
   box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
   z-index: 10;
}

.menu-item {
   display: flex;
   align-items: center;
   gap: 8px;
   width: 100%;
   padding: 8px 10px;
   font-size: 13px;
   font-weight: 500;
   color: #f5f5f7;
   border-radius: 6px;
   cursor: pointer;
   text-align: left;
}

.menu-item:hover {
   background: rgba(255, 255, 255, 0.08);
}

.menu-item.active {
   color: #0a84ff;
}

.menu-item.destructive {
   color: #ff453a;
}

.menu-item.menu-back {
   color: #98989d;
}

.menu-icon {
   width: 16px;
   height: 16px;
   flex-shrink: 0;
   display: inline-block;
}

.menu-icon.check {
   color: #0a84ff;
}

.menu-label {
   flex: 1;
}

.menu-value {
   font-size: 12px;
   color: #98989d;
}

.menu-chevron {
   width: 14px;
   height: 14px;
   color: #98989d;
   flex-shrink: 0;
}

.menu-divider {
   height: 1px;
   background: #48484a;
   margin: 4px 0;
}

/* Error */
.error {
   font-size: 12px;
   color: #ff453a;
   background: rgba(255, 69, 58, 0.1);
   padding: 8px 12px;
   border-radius: 6px;
}

/* Event log */
.event-log {
   margin-top: 16px;
   background: #1c1c1e;
   border-radius: 10px;
   overflow: hidden;
}

.event-log-header {
   display: flex;
   align-items: center;
   justify-content: space-between;
   padding: 10px 12px;
   border-bottom: 1px solid #2c2c2e;
}

.event-log-header h2 {
   font-size: 14px;
   font-weight: 600;
   margin: 0;
}

.event-log-actions {
   display: flex;
   align-items: center;
   gap: 10px;
}

.event-filter {
   font-size: 12px;
   color: #98989d;
   display: flex;
   align-items: center;
   gap: 4px;
   cursor: pointer;
}

.event-filter input {
   margin: 0;
}

.event-clear-btn {
   font-size: 11px;
   font-weight: 600;
   color: #98989d;
   padding: 2px 8px;
   border: 1px solid #48484a;
   border-radius: 4px;
   cursor: pointer;
}

.event-clear-btn:hover {
   color: #f5f5f7;
   border-color: #636366;
}

.event-log-entries {
   max-height: 260px;
   overflow-y: auto;
   font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
   font-size: 11px;
}

.event-entry {
   border-bottom: 1px solid #2c2c2e;
}

.event-entry:last-child {
   border-bottom: none;
}

.event-row {
   display: flex;
   align-items: center;
   gap: 8px;
   padding: 5px 12px;
   cursor: pointer;
   font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
   font-size: 11px;
}

.event-row:hover {
   background: rgba(255, 255, 255, 0.04);
}

.event-badge {
   font-size: 10px;
   font-weight: 600;
   padding: 2px 6px;
   border-radius: 3px;
   text-transform: uppercase;
   flex-shrink: 0;
   min-width: 56px;
   text-align: center;
}

.event-badge.state {
   background: rgba(10, 132, 255, 0.2);
   color: #0a84ff;
}

.event-badge.time {
   background: rgba(152, 152, 157, 0.15);
   color: #636366;
}

.event-summary {
   flex: 1;
   color: #d1d1d6;
   overflow: hidden;
   text-overflow: ellipsis;
   white-space: nowrap;
}

.event-time {
   color: #636366;
   flex-shrink: 0;
   font-variant-numeric: tabular-nums;
}

.event-payload {
   margin: 0;
   padding: 6px 12px 8px 32px;
   color: #98989d;
   background: rgba(0, 0, 0, 0.2);
   font-size: 10px;
   line-height: 1.5;
   overflow-x: auto;
}

.event-empty {
   padding: 24px 12px;
   text-align: center;
   color: #636366;
   font-size: 12px;
}
</style>
