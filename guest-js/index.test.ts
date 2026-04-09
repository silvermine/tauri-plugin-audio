/**
 * Sanity checks to test the bridge between TypeScript and the Tauri commands.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { getPlayer } from './index';
import {
   PlaybackStatus,
   AudioAction,
   hasAction,
   hasAnyAction,
} from './types';
import { attachPlayer } from './actions';

let lastCmd = '',
    lastArgs: Record<string, unknown> = {};

const IDLE_STATE = {
   status: PlaybackStatus.Idle,
   src: null,
   title: null,
   artist: null,
   artwork: null,
   currentTime: 0,
   duration: 0,
   volume: 1,
   muted: false,
   playbackRate: 1,
   loop: false,
   error: null,
};

const READY_STATE = {
   ...IDLE_STATE,
   status: PlaybackStatus.Ready,
   src: 'https://example.com/song.mp3',
   title: 'Test Song',
   artist: 'Test Artist',
};

const PLAYING_STATE = {
   ...READY_STATE,
   status: PlaybackStatus.Playing,
   duration: 180,
   currentTime: 42,
};

const PAUSED_STATE = {
   ...PLAYING_STATE,
   status: PlaybackStatus.Paused,
};

const ENDED_STATE = {
   ...PLAYING_STATE,
   status: PlaybackStatus.Ended,
   currentTime: 180,
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
      if (cmd === 'plugin:audio|prepare') {
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
      if (cmd === 'plugin:audio|set_volume') {
         return { ...PLAYING_STATE, volume: (args as { level: number }).level };
      }
      if (cmd === 'plugin:audio|set_muted') {
         return { ...PLAYING_STATE, muted: (args as { muted: boolean }).muted };
      }
      if (cmd === 'plugin:audio|set_playback_rate') {
         return { ...PLAYING_STATE, playbackRate: (args as { rate: number }).rate };
      }
      if (cmd === 'plugin:audio|set_loop') {
         return { ...PLAYING_STATE, loop: (args as { looping: boolean }).looping };
      }
      if (cmd === 'plugin:audio|is_native') {
         return false;
      }
      return undefined;
   });
});

afterEach(() => { return clearMocks(); });

describe('getPlayer', () => {
   it('invokes get_state and returns a player with actions attached', async () => {
      const player = await getPlayer();

      expect(lastCmd).toBe('plugin:audio|get_state');
      expect(player.status).toBe(PlaybackStatus.Idle);
      expect(hasAction(player, AudioAction.Prepare)).toBe(true);
   });
});

describe('transport actions', () => {
   it('prepare — sends src and metadata, returns Ready player', async () => {
      const player = await getPlayer();

      if (!hasAction(player, AudioAction.Prepare)) {
         throw new Error('expected prepare action');
      }
      const response = await player.prepare('https://example.com/song.mp3', {
         title: 'Test Song',
         artist: 'Test Artist',
      });

      expect(lastCmd).toBe('plugin:audio|prepare');
      expect(lastArgs.src).toBe('https://example.com/song.mp3');
      expect((lastArgs.metadata as Record<string, unknown>).title).toBe('Test Song');
      expect(response.isExpectedStatus).toBe(true);
      expect(response.player.status).toBe(PlaybackStatus.Ready);
   });

   it('play — returns Playing player', async () => {
      const ready = attachPlayer(READY_STATE);

      if (!hasAction(ready, AudioAction.Play)) {
         throw new Error('expected play action');
      }
      const response = await ready.play();

      expect(lastCmd).toBe('plugin:audio|play');
      expect(response.isExpectedStatus).toBe(true);
      expect(response.player.status).toBe(PlaybackStatus.Playing);
   });

   it('pause — returns Paused player', async () => {
      const playing = attachPlayer(PLAYING_STATE);

      if (!hasAction(playing, AudioAction.Pause)) {
         throw new Error('expected pause action');
      }
      const response = await playing.pause();

      expect(lastCmd).toBe('plugin:audio|pause');
      expect(response.isExpectedStatus).toBe(true);
      expect(response.player.status).toBe(PlaybackStatus.Paused);
   });

   it('stop — returns Idle player', async () => {
      const playing = attachPlayer(PLAYING_STATE);

      if (!hasAction(playing, AudioAction.Stop)) {
         throw new Error('expected stop action');
      }
      const response = await playing.stop();

      expect(lastCmd).toBe('plugin:audio|stop');
      expect(response.isExpectedStatus).toBe(true);
      expect(response.player.status).toBe(PlaybackStatus.Idle);
   });

   it('seek — sends position, returns player at new time', async () => {
      const playing = attachPlayer(PLAYING_STATE);

      if (!hasAction(playing, AudioAction.Seek)) {
         throw new Error('expected seek action');
      }
      const response = await playing.seek(90);

      expect(lastCmd).toBe('plugin:audio|seek');
      expect(lastArgs.position).toBe(90);
      expect(response.isExpectedStatus).toBe(true);
      expect(response.player.currentTime).toBe(90);
   });

   it('handles errors thrown by the backend', async () => {
      mockIPC(() => { throw new Error('audio error'); });

      const player = await getPlayer().catch(() => {
         return attachPlayer(IDLE_STATE);
      });

      if (!hasAction(player, AudioAction.Prepare)) {
         throw new Error('expected prepare action');
      }
      await expect(player.prepare('test.mp3')).rejects.toThrow('audio error');
   });
});

describe('player controls (always available)', () => {
   it('setVolume — sends level, returns updated player', async () => {
      const player = attachPlayer(PLAYING_STATE);

      const updated = await player.setVolume(0.5);

      expect(lastCmd).toBe('plugin:audio|set_volume');
      expect(lastArgs.level).toBe(0.5);
      expect(updated.volume).toBe(0.5);
   });

   it('setMuted — sends muted flag, returns updated player', async () => {
      const player = attachPlayer(PLAYING_STATE);

      const updated = await player.setMuted(true);

      expect(lastCmd).toBe('plugin:audio|set_muted');
      expect(lastArgs.muted).toBe(true);
      expect(updated.muted).toBe(true);
   });

   it('setPlaybackRate — sends rate, returns updated player', async () => {
      const player = attachPlayer(PLAYING_STATE);

      const updated = await player.setPlaybackRate(2.0);

      expect(lastCmd).toBe('plugin:audio|set_playback_rate');
      expect(lastArgs.rate).toBe(2.0);
      expect(updated.playbackRate).toBe(2.0);
   });

   it('setLoop — sends looping flag, returns updated player', async () => {
      const player = attachPlayer(PLAYING_STATE);

      const updated = await player.setLoop(true);

      expect(lastCmd).toBe('plugin:audio|set_loop');
      expect(lastArgs.looping).toBe(true);
      expect(updated.loop).toBe(true);
   });

   it('controls are available in Idle state', () => {
      const player = attachPlayer(IDLE_STATE);

      expect(typeof player.setVolume).toBe('function');
      expect(typeof player.setMuted).toBe('function');
      expect(typeof player.setPlaybackRate).toBe('function');
      expect(typeof player.setLoop).toBe('function');
      expect(typeof player.listen).toBe('function');
      expect(typeof player.onTimeUpdate).toBe('function');
   });

   it('controls are available in Error state', () => {
      const player = attachPlayer({ ...IDLE_STATE, status: PlaybackStatus.Error, error: 'fail' });

      expect(typeof player.setVolume).toBe('function');
      expect(typeof player.setMuted).toBe('function');
      expect(typeof player.setPlaybackRate).toBe('function');
      expect(typeof player.setLoop).toBe('function');
      expect(typeof player.listen).toBe('function');
      expect(typeof player.onTimeUpdate).toBe('function');
   });
});

describe('state machine — action availability', () => {
   it('Idle: only prepare is available', () => {
      const player = attachPlayer(IDLE_STATE);

      expect(hasAction(player, AudioAction.Prepare)).toBe(true);
      expect(hasAction(player, AudioAction.Play)).toBe(false);
      expect(hasAction(player, AudioAction.Pause)).toBe(false);
      expect(hasAction(player, AudioAction.Stop)).toBe(false);
      expect(hasAction(player, AudioAction.Seek)).toBe(false);
   });

   it('Loading: only stop is available', () => {
      const player = attachPlayer({ ...IDLE_STATE, status: PlaybackStatus.Loading });

      expect(hasAction(player, AudioAction.Stop)).toBe(true);
      expect(hasAction(player, AudioAction.Prepare)).toBe(false);
      expect(hasAction(player, AudioAction.Play)).toBe(false);
      expect(hasAction(player, AudioAction.Pause)).toBe(false);
   });

   it('Ready: play, seek, and stop are available', () => {
      const player = attachPlayer(READY_STATE);

      expect(hasAction(player, AudioAction.Play)).toBe(true);
      expect(hasAction(player, AudioAction.Seek)).toBe(true);
      expect(hasAction(player, AudioAction.Stop)).toBe(true);
      expect(hasAction(player, AudioAction.Pause)).toBe(false);
      expect(hasAction(player, AudioAction.Prepare)).toBe(false);
   });

   it('Playing: pause, seek, and stop are available', () => {
      const player = attachPlayer(PLAYING_STATE);

      expect(hasAction(player, AudioAction.Pause)).toBe(true);
      expect(hasAction(player, AudioAction.Seek)).toBe(true);
      expect(hasAction(player, AudioAction.Stop)).toBe(true);
      expect(hasAction(player, AudioAction.Play)).toBe(false);
      expect(hasAction(player, AudioAction.Prepare)).toBe(false);
   });

   it('Paused: play, seek, and stop are available', () => {
      const player = attachPlayer(PAUSED_STATE);

      expect(hasAction(player, AudioAction.Play)).toBe(true);
      expect(hasAction(player, AudioAction.Seek)).toBe(true);
      expect(hasAction(player, AudioAction.Stop)).toBe(true);
      expect(hasAction(player, AudioAction.Pause)).toBe(false);
      expect(hasAction(player, AudioAction.Prepare)).toBe(false);
   });

   it('Ended: play, seek, prepare, and stop are available', () => {
      const player = attachPlayer(ENDED_STATE);

      expect(hasAction(player, AudioAction.Play)).toBe(true);
      expect(hasAction(player, AudioAction.Seek)).toBe(true);
      expect(hasAction(player, AudioAction.Prepare)).toBe(true);
      expect(hasAction(player, AudioAction.Stop)).toBe(true);
      expect(hasAction(player, AudioAction.Pause)).toBe(false);
   });

   it('Error: only prepare is available', () => {
      const player = attachPlayer({ ...IDLE_STATE, status: PlaybackStatus.Error, error: 'fail' });

      expect(hasAction(player, AudioAction.Prepare)).toBe(true);
      expect(hasAction(player, AudioAction.Play)).toBe(false);
      expect(hasAction(player, AudioAction.Pause)).toBe(false);
      expect(hasAction(player, AudioAction.Stop)).toBe(false);
      expect(hasAction(player, AudioAction.Seek)).toBe(false);
   });

   it('hasAnyAction returns true for states with transport actions', () => {
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

      expect(typeof (idle as unknown as Record<string, unknown>).prepare).toBe('function');
      expect(typeof (idle as unknown as Record<string, unknown>).play).toBe('undefined');
      expect(typeof (idle as unknown as Record<string, unknown>).pause).toBe('undefined');
   });

   it('preserves all state fields on the returned object', () => {
      const player = attachPlayer(PLAYING_STATE);

      expect(player.status).toBe(PlaybackStatus.Playing);
      expect(player.src).toBe('https://example.com/song.mp3');
      expect(player.title).toBe('Test Song');
      expect(player.artist).toBe('Test Artist');
      expect(player.currentTime).toBe(42);
      expect(player.duration).toBe(180);
      expect(player.volume).toBe(1);
      expect(player.muted).toBe(false);
      expect(player.playbackRate).toBe(1);
      expect(player.loop).toBe(false);
      expect(player.error).toBeNull();
   });
});
