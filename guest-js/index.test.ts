/**
 * Sanity checks to test the bridge between TypeScript and the Tauri commands.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';

// Stub `@tauri-apps/api/event::listen` so tests can capture registered
// handlers and fire synthetic events at them, exercising both the
// per-channel routing and the `PluginEventManager` lifecycle.
const eventState = vi.hoisted(() => {
   return {
      listeners: new Map<string, Set<(event: { payload: unknown }) => void>>(),
      listenCalls: 0,
   };
});

vi.mock('@tauri-apps/api/event', () => {
   return {
      listen: async (
         eventName: string,
         handler: (event: { payload: unknown }) => void
      ): Promise<() => void> => {
         eventState.listenCalls += 1;
         const set = eventState.listeners.get(eventName) ?? new Set();

         set.add(handler);
         eventState.listeners.set(eventName, set);
         return () => { set.delete(handler); };
      },
   };
});

import { getPlayer } from './index';
import {
   PlaybackStatus,
   AudioAction,
   LoopMode,
   PlaylistItem,
   SettingsChange,
   StateChange,
   TimeUpdate,
   TrackChange,
   hasAction,
   hasAnyAction,
} from './types';
import { attachPlayer } from './actions';

function fireEvent(eventName: string, payload: unknown): void {
   eventState.listeners.get(eventName)?.forEach((h) => { h({ payload }); });
}

function resetEventMock(): void {
   eventState.listeners.clear();
   eventState.listenCalls = 0;
}

/** Placeholder listener used when a test only cares about subscription
 * lifecycle counts and not the event payloads themselves. */
function noop(): void {
   // intentionally empty
}

let lastCmd = '',
    lastArgs: Record<string, unknown> = {};

const PLAYLIST: PlaylistItem[] = [
   { src: 'https://example.com/track-1.mp3', metadata: { title: 'Track 1', artist: 'Artist' } },
   { src: 'https://example.com/track-2.mp3', metadata: { title: 'Track 2', artist: 'Artist' } },
   { src: 'https://example.com/track-3.mp3', metadata: { title: 'Track 3', artist: 'Artist' } },
];

const IDLE_STATE = {
   status: PlaybackStatus.Idle,
   playlist: [],
   currentIndex: null,
   currentTime: 0,
   duration: 0,
   volume: 1,
   muted: false,
   playbackRate: 1,
   loopMode: LoopMode.Off,
   error: null,
};

const READY_STATE = {
   ...IDLE_STATE,
   status: PlaybackStatus.Ready,
   playlist: PLAYLIST,
   currentIndex: 0,
   duration: 180,
};

const PLAYING_STATE = {
   ...READY_STATE,
   status: PlaybackStatus.Playing,
   currentTime: 42,
};

const PLAYING_TRACK_2_STATE = {
   ...PLAYING_STATE,
   currentIndex: 1,
   currentTime: 0,
};

const PAUSED_STATE = {
   ...PLAYING_STATE,
   status: PlaybackStatus.Paused,
};

const ENDED_STATE = {
   ...PLAYING_STATE,
   status: PlaybackStatus.Ended,
   currentTime: 180,
   currentIndex: PLAYLIST.length - 1,
};

const ACTION_RESPONSE_BASE = {
   isExpectedStatus: true,
};

beforeEach(() => {
   mockIPC((cmd, args) => {
      lastCmd = cmd;
      lastArgs = args as Record<string, unknown>;

      if (cmd === 'plugin:audio|get_state') {
         return IDLE_STATE;
      }
      if (cmd === 'plugin:audio|load') {
         return {
            ...ACTION_RESPONSE_BASE,
            expectedStatus: PlaybackStatus.Ready,
            player: READY_STATE,
         };
      }
      if (cmd === 'plugin:audio|play') {
         return {
            ...ACTION_RESPONSE_BASE,
            expectedStatus: PlaybackStatus.Playing,
            player: PLAYING_STATE,
         };
      }
      if (cmd === 'plugin:audio|pause') {
         return {
            ...ACTION_RESPONSE_BASE,
            expectedStatus: PlaybackStatus.Paused,
            player: PAUSED_STATE,
         };
      }
      if (cmd === 'plugin:audio|stop') {
         return {
            ...ACTION_RESPONSE_BASE,
            expectedStatus: PlaybackStatus.Idle,
            player: IDLE_STATE,
         };
      }
      if (cmd === 'plugin:audio|seek') {
         return {
            ...ACTION_RESPONSE_BASE,
            expectedStatus: PlaybackStatus.Playing,
            player: { ...PLAYING_STATE, currentTime: (args as { position: number }).position },
         };
      }
      if (cmd === 'plugin:audio|next') {
         return {
            ...ACTION_RESPONSE_BASE,
            expectedStatus: PlaybackStatus.Ready,
            player: PLAYING_TRACK_2_STATE,
         };
      }
      if (cmd === 'plugin:audio|prev') {
         return {
            ...ACTION_RESPONSE_BASE,
            expectedStatus: PlaybackStatus.Ready,
            player: PLAYING_STATE,
         };
      }
      if (cmd === 'plugin:audio|jump_to') {
         const targetIdx = (args as { index: number }).index;

         return {
            ...ACTION_RESPONSE_BASE,
            expectedStatus: PlaybackStatus.Ready,
            player: { ...PLAYING_STATE, currentIndex: targetIdx },
         };
      }
      if (cmd === 'plugin:audio|set_volume') {
         return { ...PLAYING_STATE, volume: (args as { level: number }).level };
      }
      if (cmd === 'plugin:audio|set_muted') {
         return { ...PLAYING_STATE, muted: (args as { muted: boolean }).muted };
      }
      if (cmd === 'plugin:audio|set_playback_rate') {
         return { ...PLAYING_STATE, playbackRate: (args as { rate: number }).rate };
      }
      if (cmd === 'plugin:audio|set_loop_mode') {
         return { ...PLAYING_STATE, loopMode: (args as { mode: LoopMode }).mode };
      }
      if (cmd === 'plugin:audio|is_native') {
         return false;
      }
      return undefined;
   });
});

afterEach(() => {
   resetEventMock();
   return clearMocks();
});

describe('getPlayer', () => {
   it('invokes get_state and returns a player with actions attached', async () => {
      const player = await getPlayer();

      expect(lastCmd).toBe('plugin:audio|get_state');
      expect(player.status).toBe(PlaybackStatus.Idle);
      expect(hasAction(player, AudioAction.Load)).toBe(true);
   });
});

describe('transport actions', () => {
   it('load — sends playlist and startIndex, returns Ready player', async () => {
      const player = await getPlayer();

      if (!hasAction(player, AudioAction.Load)) {
         throw new Error('expected load action');
      }
      const response = await player.load(PLAYLIST, 0);

      expect(lastCmd).toBe('plugin:audio|load');
      expect((lastArgs.playlist as PlaylistItem[]).length).toBe(3);
      expect((lastArgs.playlist as PlaylistItem[])[0].src).toBe('https://example.com/track-1.mp3');
      expect(lastArgs.startIndex).toBe(0);
      expect(response.isExpectedStatus).toBe(true);
      expect(response.player.status).toBe(PlaybackStatus.Ready);
      expect(response.player.playlist.length).toBe(3);
      expect(response.player.currentIndex).toBe(0);
   });

   it('load — single-track callers pass a one-item array', async () => {
      const player = await getPlayer();

      if (!hasAction(player, AudioAction.Load)) {
         throw new Error('expected load action');
      }
      await player.load([ { src: 'https://example.com/song.mp3' } ]);

      expect(lastCmd).toBe('plugin:audio|load');
      expect((lastArgs.playlist as PlaylistItem[]).length).toBe(1);
   });

   it('play — returns Playing player', async () => {
      const ready = attachPlayer(READY_STATE);

      if (!hasAction(ready, AudioAction.Play)) {
         throw new Error('expected play action');
      }
      const response = await ready.play();

      expect(lastCmd).toBe('plugin:audio|play');
      expect(response.player.status).toBe(PlaybackStatus.Playing);
   });

   it('pause — returns Paused player', async () => {
      const playing = attachPlayer(PLAYING_STATE);

      if (!hasAction(playing, AudioAction.Pause)) {
         throw new Error('expected pause action');
      }
      const response = await playing.pause();

      expect(response.player.status).toBe(PlaybackStatus.Paused);
   });

   it('stop — returns Idle player', async () => {
      const playing = attachPlayer(PLAYING_STATE);

      if (!hasAction(playing, AudioAction.Stop)) {
         throw new Error('expected stop action');
      }
      const response = await playing.stop();

      expect(response.player.status).toBe(PlaybackStatus.Idle);
   });

   it('seek — sends position, returns player at new time', async () => {
      const playing = attachPlayer(PLAYING_STATE);

      if (!hasAction(playing, AudioAction.Seek)) {
         throw new Error('expected seek action');
      }
      const response = await playing.seek(90);

      expect(lastArgs.position).toBe(90);
      expect(response.player.currentTime).toBe(90);
   });

   it('next — advances to next playlist item', async () => {
      const playing = attachPlayer(PLAYING_STATE);

      if (!hasAction(playing, AudioAction.Next)) {
         throw new Error('expected next action');
      }
      const response = await playing.next();

      expect(lastCmd).toBe('plugin:audio|next');
      expect(response.player.currentIndex).toBe(1);
   });

   it('prev — moves to previous playlist item', async () => {
      const playing = attachPlayer(PLAYING_TRACK_2_STATE);

      if (!hasAction(playing, AudioAction.Prev)) {
         throw new Error('expected prev action');
      }
      const response = await playing.prev();

      expect(lastCmd).toBe('plugin:audio|prev');
      expect(response.player.currentIndex).toBe(0);
   });

   it('jumpTo — sends snake_case command and target index', async () => {
      const playing = attachPlayer(PLAYING_STATE);

      if (!hasAction(playing, AudioAction.JumpTo)) {
         throw new Error('expected jumpTo action');
      }
      const response = await playing.jumpTo(2);

      expect(lastCmd).toBe('plugin:audio|jump_to');
      expect(lastArgs.index).toBe(2);
      expect(response.player.currentIndex).toBe(2);
   });

   it('prev — from Ended (RestartCurrent path) preserves Ended status', async () => {
      mockIPC((cmd) => {
         lastCmd = cmd;
         if (cmd === 'plugin:audio|prev') {
            return {
               ...ACTION_RESPONSE_BASE,
               expectedStatus: PlaybackStatus.Ended,
               player: { ...ENDED_STATE, currentTime: 0 },
            };
         }
         return undefined;
      });

      const ended = attachPlayer(ENDED_STATE);

      if (!hasAction(ended, AudioAction.Prev)) {
         throw new Error('expected prev action');
      }
      const response = await ended.prev();

      expect(response.player.status).toBe(PlaybackStatus.Ended);
      expect(response.isExpectedStatus).toBe(true);
   });

   it('jumpTo — to current index from Ended preserves Ended status', async () => {
      mockIPC((cmd, args) => {
         lastCmd = cmd;
         lastArgs = args as Record<string, unknown>;
         if (cmd === 'plugin:audio|jump_to') {
            return {
               ...ACTION_RESPONSE_BASE,
               expectedStatus: PlaybackStatus.Ended,
               player: { ...ENDED_STATE, currentTime: 0 },
            };
         }
         return undefined;
      });

      const ended = attachPlayer(ENDED_STATE);

      if (!hasAction(ended, AudioAction.JumpTo)) {
         throw new Error('expected jumpTo action');
      }
      const response = await ended.jumpTo(ENDED_STATE.currentIndex);

      expect(response.player.status).toBe(PlaybackStatus.Ended);
      expect(response.isExpectedStatus).toBe(true);
   });

   it('prev — from Paused preserves Paused status', async () => {
      mockIPC((cmd, args) => {
         lastCmd = cmd;
         lastArgs = args as Record<string, unknown>;
         if (cmd === 'plugin:audio|prev') {
            return {
               ...ACTION_RESPONSE_BASE,
               expectedStatus: PlaybackStatus.Paused,
               player: { ...PAUSED_STATE, currentIndex: 0, currentTime: 0 },
            };
         }
         return undefined;
      });

      const paused = attachPlayer({ ...PAUSED_STATE, currentIndex: 1 });

      if (!hasAction(paused, AudioAction.Prev)) {
         throw new Error('expected prev action');
      }
      const response = await paused.prev();

      expect(response.player.status).toBe(PlaybackStatus.Paused);
      expect(response.player.currentIndex).toBe(0);
      expect(response.isExpectedStatus).toBe(true);
   });

   it('handles errors thrown by the backend', async () => {
      mockIPC(() => { throw new Error('audio error'); });

      const player = await getPlayer().catch(() => {
         return attachPlayer(IDLE_STATE);
      });

      if (!hasAction(player, AudioAction.Load)) {
         throw new Error('expected load action');
      }
      await expect(player.load([ { src: 'test.mp3' } ])).rejects.toThrow('audio error');
   });
});

describe('player controls (always available)', () => {
   it('setVolume — sends level, returns updated player', async () => {
      const player = attachPlayer(PLAYING_STATE);

      const updated = await player.setVolume(0.5);

      expect(lastArgs.level).toBe(0.5);
      expect(updated.volume).toBe(0.5);
   });

   it('setMuted — sends muted flag, returns updated player', async () => {
      const player = attachPlayer(PLAYING_STATE);

      const updated = await player.setMuted(true);

      expect(lastArgs.muted).toBe(true);
      expect(updated.muted).toBe(true);
   });

   it('setPlaybackRate — sends rate, returns updated player', async () => {
      const player = attachPlayer(PLAYING_STATE);

      const updated = await player.setPlaybackRate(2.0);

      expect(lastArgs.rate).toBe(2.0);
      expect(updated.playbackRate).toBe(2.0);
   });

   it('setLoopMode — sends mode, returns updated player', async () => {
      const player = attachPlayer(PLAYING_STATE);

      const updatedAll = await player.setLoopMode(LoopMode.All);

      expect(lastCmd).toBe('plugin:audio|set_loop_mode');
      expect(lastArgs.mode).toBe(LoopMode.All);
      expect(updatedAll.loopMode).toBe(LoopMode.All);

      const updatedOne = await player.setLoopMode(LoopMode.One);

      expect(lastArgs.mode).toBe(LoopMode.One);
      expect(updatedOne.loopMode).toBe(LoopMode.One);

      const updatedOff = await player.setLoopMode(LoopMode.Off);

      expect(lastArgs.mode).toBe(LoopMode.Off);
      expect(updatedOff.loopMode).toBe(LoopMode.Off);
   });

   it('controls are available in Idle state', () => {
      const player = attachPlayer(IDLE_STATE);

      expect(typeof player.setVolume).toBe('function');
      expect(typeof player.setMuted).toBe('function');
      expect(typeof player.setPlaybackRate).toBe('function');
      expect(typeof player.setLoopMode).toBe('function');
      expect(typeof player.onStateChanged).toBe('function');
      expect(typeof player.onTrackChanged).toBe('function');
      expect(typeof player.onSettingsChanged).toBe('function');
      expect(typeof player.onTimeUpdate).toBe('function');
   });

   it('controls are available in Error state', () => {
      const player = attachPlayer({ ...IDLE_STATE, status: PlaybackStatus.Error, error: 'fail' });

      expect(typeof player.setVolume).toBe('function');
      expect(typeof player.setLoopMode).toBe('function');
   });
});

describe('state machine — action availability', () => {
   it('Idle: only load is available', () => {
      const player = attachPlayer(IDLE_STATE);

      expect(hasAction(player, AudioAction.Load)).toBe(true);
      expect(hasAction(player, AudioAction.Play)).toBe(false);
      expect(hasAction(player, AudioAction.Next)).toBe(false);
      expect(hasAction(player, AudioAction.Prev)).toBe(false);
   });

   it('Loading: only stop is available', () => {
      const player = attachPlayer({ ...IDLE_STATE, status: PlaybackStatus.Loading });

      expect(hasAction(player, AudioAction.Stop)).toBe(true);
      expect(hasAction(player, AudioAction.Next)).toBe(false);
      expect(hasAction(player, AudioAction.Prev)).toBe(false);
   });

   it('Ready: play, seek, stop, next, and prev are available', () => {
      const player = attachPlayer(READY_STATE);

      expect(hasAction(player, AudioAction.Play)).toBe(true);
      expect(hasAction(player, AudioAction.Seek)).toBe(true);
      expect(hasAction(player, AudioAction.Stop)).toBe(true);
      expect(hasAction(player, AudioAction.Next)).toBe(true);
      expect(hasAction(player, AudioAction.Prev)).toBe(true);
      expect(hasAction(player, AudioAction.Pause)).toBe(false);
   });

   it('Playing: pause, seek, stop, next, and prev are available', () => {
      const player = attachPlayer(PLAYING_STATE);

      expect(hasAction(player, AudioAction.Pause)).toBe(true);
      expect(hasAction(player, AudioAction.Seek)).toBe(true);
      expect(hasAction(player, AudioAction.Stop)).toBe(true);
      expect(hasAction(player, AudioAction.Next)).toBe(true);
      expect(hasAction(player, AudioAction.Prev)).toBe(true);
      expect(hasAction(player, AudioAction.JumpTo)).toBe(true);
   });

   it('Paused: play, seek, stop, next, and prev are available', () => {
      const player = attachPlayer(PAUSED_STATE);

      expect(hasAction(player, AudioAction.Play)).toBe(true);
      expect(hasAction(player, AudioAction.Next)).toBe(true);
      expect(hasAction(player, AudioAction.Prev)).toBe(true);
   });

   it('Ended: play, seek, load, stop, next, and prev are available', () => {
      const player = attachPlayer(ENDED_STATE);

      expect(hasAction(player, AudioAction.Play)).toBe(true);
      expect(hasAction(player, AudioAction.Load)).toBe(true);
      expect(hasAction(player, AudioAction.Next)).toBe(true);
      expect(hasAction(player, AudioAction.Prev)).toBe(true);
   });

   it('Error: load, stop, next, prev, and jumpTo are available', () => {
      const player = attachPlayer({ ...IDLE_STATE, status: PlaybackStatus.Error, error: 'fail' });

      expect(hasAction(player, AudioAction.Load)).toBe(true);
      expect(hasAction(player, AudioAction.Stop)).toBe(true);
      expect(hasAction(player, AudioAction.Next)).toBe(true);
      expect(hasAction(player, AudioAction.Prev)).toBe(true);
      expect(hasAction(player, AudioAction.JumpTo)).toBe(true);
      expect(hasAction(player, AudioAction.Play)).toBe(false);
      expect(hasAction(player, AudioAction.Pause)).toBe(false);
   });

   it('hasAnyAction returns true for all states', () => {
      expect(hasAnyAction(attachPlayer(IDLE_STATE))).toBe(true);
      expect(hasAnyAction(attachPlayer(READY_STATE))).toBe(true);
      expect(hasAnyAction(attachPlayer(PLAYING_STATE))).toBe(true);
      expect(hasAnyAction(attachPlayer(PAUSED_STATE))).toBe(true);
      expect(hasAnyAction(attachPlayer(ENDED_STATE))).toBe(true);
      expect(hasAnyAction(attachPlayer({ ...IDLE_STATE, status: PlaybackStatus.Error }))).toBe(true);
      expect(hasAnyAction(attachPlayer({ ...IDLE_STATE, status: PlaybackStatus.Loading }))).toBe(true);
   });

   it('attaches only the allowed transport methods as callable functions', () => {
      const idle = attachPlayer(IDLE_STATE);

      expect(typeof (idle as unknown as Record<string, unknown>).load).toBe('function');
      expect(typeof (idle as unknown as Record<string, unknown>).play).toBe('undefined');
      expect(typeof (idle as unknown as Record<string, unknown>).next).toBe('undefined');
      expect(typeof (idle as unknown as Record<string, unknown>).prev).toBe('undefined');
   });

   it('preserves all state fields on the returned object', () => {
      const player = attachPlayer(PLAYING_STATE);

      expect(player.status).toBe(PlaybackStatus.Playing);
      expect(player.playlist.length).toBe(3);
      expect(player.currentIndex).toBe(0);
      expect(player.currentTime).toBe(42);
      expect(player.duration).toBe(180);
      expect(player.volume).toBe(1);
      expect(player.muted).toBe(false);
      expect(player.playbackRate).toBe(1);
      expect(player.loopMode).toBe(LoopMode.Off);
      expect(player.error).toBeNull();
   });
});

describe('event channel routing', () => {
   it('onStateChanged receives state-changed payloads', async () => {
      const events: StateChange[] = [],
            player = await getPlayer(),
            unlisten = await player.onStateChanged((change) => { events.push(change); });

      fireEvent('tauri-plugin-audio:state-changed', { status: PlaybackStatus.Playing, error: null });

      expect(events).toHaveLength(1);
      expect(events[0]).toEqual({ status: PlaybackStatus.Playing, error: null });

      unlisten();
   });

   it('onTrackChanged receives track-changed payloads', async () => {
      const events: TrackChange[] = [],
            player = await getPlayer(),
            unlisten = await player.onTrackChanged((change) => { events.push(change); }),
            item: PlaylistItem = { src: 'a.mp3', metadata: { title: 'A' } };

      fireEvent('tauri-plugin-audio:track-changed', {
         currentIndex: 1,
         duration: 120,
         item,
      });

      expect(events).toHaveLength(1);
      expect(events[0].currentIndex).toBe(1);
      expect(events[0].duration).toBe(120);
      expect(events[0].item.src).toBe('a.mp3');

      unlisten();
   });

   it('onSettingsChanged receives partial settings deltas', async () => {
      const events: SettingsChange[] = [],
            player = await getPlayer(),
            unlisten = await player.onSettingsChanged((change) => { events.push(change); });

      fireEvent('tauri-plugin-audio:settings-changed', { volume: 0.5 });
      fireEvent('tauri-plugin-audio:settings-changed', { muted: true });

      expect(events).toEqual([ { volume: 0.5 }, { muted: true } ]);

      unlisten();
   });

   it('onTimeUpdate receives time-update payloads', async () => {
      const events: TimeUpdate[] = [],
            player = await getPlayer(),
            unlisten = await player.onTimeUpdate((time) => { events.push(time); });

      fireEvent('tauri-plugin-audio:time-update', { currentTime: 42, duration: 120 });

      expect(events).toEqual([ { currentTime: 42, duration: 120 } ]);

      unlisten();
   });

   it('events on one channel do not leak to other channels', async () => {
      const stateEvents: StateChange[] = [],
            trackEvents: TrackChange[] = [],
            settingsEvents: SettingsChange[] = [],
            timeEvents: TimeUpdate[] = [],
            player = await getPlayer(),
            unlistenState = await player.onStateChanged((c) => { stateEvents.push(c); }),
            unlistenTrack = await player.onTrackChanged((c) => { trackEvents.push(c); }),
            unlistenSettings = await player.onSettingsChanged((c) => { settingsEvents.push(c); }),
            unlistenTime = await player.onTimeUpdate((c) => { timeEvents.push(c); });

      fireEvent('tauri-plugin-audio:settings-changed', { volume: 0.7 });

      expect(settingsEvents).toHaveLength(1);
      expect(stateEvents).toHaveLength(0);
      expect(trackEvents).toHaveLength(0);
      expect(timeEvents).toHaveLength(0);

      unlistenState();
      unlistenTrack();
      unlistenSettings();
      unlistenTime();
   });

   it('emits the expected sequence for a navigation: state Loading → state Ready → track', async () => {
      const channelLog: string[] = [],
            player = await getPlayer(),
            unlistenState = await player.onStateChanged((c) => { channelLog.push(`state:${c.status}`); }),
            unlistenTrack = await player.onTrackChanged(() => { channelLog.push('track'); }),
            item: PlaylistItem = { src: 'b.mp3' };

      fireEvent('tauri-plugin-audio:state-changed', { status: PlaybackStatus.Loading, error: null });
      fireEvent('tauri-plugin-audio:state-changed', { status: PlaybackStatus.Ready, error: null });
      fireEvent('tauri-plugin-audio:track-changed', { currentIndex: 1, duration: 90, item });

      expect(channelLog).toEqual([ 'state:loading', 'state:ready', 'track' ]);

      unlistenState();
      unlistenTrack();
   });
});

describe('PluginEventManager lifecycle', () => {
   it('reuses a single global listener across multiple subscribers', async () => {
      const player = await getPlayer(),
            before = eventState.listenCalls,
            unlisten1 = await player.onStateChanged(noop),
            unlisten2 = await player.onStateChanged(noop),
            unlisten3 = await player.onStateChanged(noop);

      // All three subscribers share one underlying `listen()` call.
      expect(eventState.listenCalls - before).toBe(1);

      unlisten1();
      unlisten2();
      unlisten3();
   });

   it('fans out a single event to multiple subscribers', async () => {
      let aCalls = 0,
          bCalls = 0;

      const player = await getPlayer(),
            unlisten1 = await player.onStateChanged(() => { aCalls += 1; }),
            unlisten2 = await player.onStateChanged(() => { bCalls += 1; });

      fireEvent('tauri-plugin-audio:state-changed', { status: PlaybackStatus.Playing, error: null });

      expect(aCalls).toBe(1);
      expect(bCalls).toBe(1);

      unlisten1();
      unlisten2();
   });

   it('tears down the global listener when the last subscriber unsubscribes', async () => {
      const player = await getPlayer(),
            before = eventState.listenCalls,
            unlisten1 = await player.onStateChanged(noop);

      // Tear down — next subscribe should re-`listen()`.
      unlisten1();

      const unlisten2 = await player.onStateChanged(noop);

      expect(eventState.listenCalls - before).toBe(2);

      unlisten2();
   });

   it('only tears down once all subscribers have unsubscribed', async () => {
      const events: StateChange[] = [],
            player = await getPlayer(),
            unlisten1 = await player.onStateChanged(noop),
            unlisten2 = await player.onStateChanged((c) => { events.push(c); });

      // Drop one subscriber. The other should still receive events.
      unlisten1();

      fireEvent('tauri-plugin-audio:state-changed', { status: PlaybackStatus.Paused, error: null });

      expect(events).toHaveLength(1);

      unlisten2();
   });

   it('dedupes concurrent addListener calls during pending setup', async () => {
      // Two `addListener` calls in flight before the first one's setup
      // promise resolves should share the single `_pendingSetup`.
      const player = await getPlayer(),
            before = eventState.listenCalls,
            pending = [ player.onStateChanged(noop), player.onStateChanged(noop) ],
            [ unlisten1, unlisten2 ] = await Promise.all(pending);

      expect(eventState.listenCalls - before).toBe(1);

      unlisten1();
      unlisten2();
   });
});
