import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { addPluginListener, invoke } from '@tauri-apps/api/core';
import {
   AllAudioActions, AudioAction, AudioActionResponse, LoopMode,
   Player, PlayerControls, PlayerState, PlaybackStatus, PlaylistItem,
   PlayerWithAnyStatus, SettingsChange, StateChange, TimeUpdate,
   TrackChange, allowedActions,
} from './types';

/**
 * Generic manager for plugin event subscriptions. Handles lazy setup/teardown
 * of the global listener (native plugin or Tauri event) and dispatches
 * transformed payloads to registered listeners.
 *
 * @typeParam TRaw - The raw event payload type from the plugin.
 * @typeParam TOut - The transformed type dispatched to listeners.
 */
class PluginEventManager<TRaw, TOut> {
   private _listeners: Set<(value: TOut) => void> = new Set();
   private _eventUnlistenFn: UnlistenFn | null = null;
   private _pluginListener: { unregister: () => void } | null = null;
   private _pendingSetup: Promise<void> | null = null;
   private readonly _pluginEvent: string;
   private readonly _tauriEvent: string;
   private readonly _transform: (raw: TRaw) => TOut;

   public constructor(
      pluginEvent: string,
      tauriEvent: string,
      transform: (raw: TRaw) => TOut
   ) {
      this._pluginEvent = pluginEvent;
      this._tauriEvent = tauriEvent;
      this._transform = transform;
   }

   public async addListener(listener: (value: TOut) => void): Promise<() => void> {
      await this._ensureGlobalListener();
      this._listeners.add(listener);

      return () => {
         this._listeners.delete(listener);
         this._cleanupGlobalListener();
      };
   }

   private _ensureGlobalListener(): Promise<void> {
      if (this._eventUnlistenFn || this._pluginListener) {
         return Promise.resolve();
      }

      if (!this._pendingSetup) {
         this._pendingSetup = this._setupGlobalListener().finally(() => {
            this._pendingSetup = null;
         });
      }

      return this._pendingSetup;
   }

   private async _setupGlobalListener(): Promise<void> {
      const isNative = await invoke<boolean>('plugin:audio|is_native');

      if (isNative) {
         this._pluginListener = await addPluginListener(
            'audio',
            this._pluginEvent,
            (event: TRaw) => {
               this._notifyListeners(event);
            }
         );
      } else {
         this._eventUnlistenFn = await listen<TRaw>(
            this._tauriEvent,
            (event) => {
               this._notifyListeners(event.payload);
            }
         );
      }
   }

   private _notifyListeners(raw: TRaw): void {
      const value = this._transform(raw);

      this._listeners.forEach((listener) => { listener(value); });
   }

   private _cleanupGlobalListener(): void {
      if (this._listeners.size > 0) {
         return;
      }

      if (this._eventUnlistenFn) {
         this._eventUnlistenFn();
         this._eventUnlistenFn = null;
      }

      if (this._pluginListener) {
         this._pluginListener.unregister();
         this._pluginListener = null;
      }
   }
}

/** State-machine transitions (status + error). */
const stateEventManager = new PluginEventManager<StateChange, StateChange>(
   'state-changed',
   'tauri-plugin-audio:state-changed',
   (event) => { return event; }
);

/** Active-track changes (carries the enriched playlist item). */
const trackEventManager = new PluginEventManager<TrackChange, TrackChange>(
   'track-changed',
   'tauri-plugin-audio:track-changed',
   (event) => { return event; }
);

/** Settings deltas (only the changed field is set). */
const settingsEventManager = new PluginEventManager<SettingsChange, SettingsChange>(
   'settings-changed',
   'tauri-plugin-audio:settings-changed',
   (event) => { return event; }
);

/** High-frequency time-update events (~250 ms during playback) plus seek hits. */
const timeUpdateEventManager = new PluginEventManager<TimeUpdate, TimeUpdate>(
   'time-update',
   'tauri-plugin-audio:time-update',
   (event) => { return event; }
);

/**
 * Maps each {@link AudioAction} to the snake_case Tauri command name
 * registered in the plugin's permissions.
 */
const actionCommands: Record<AudioAction, string> = {
   [AudioAction.Load]: 'load',
   [AudioAction.Play]: 'play',
   [AudioAction.Pause]: 'pause',
   [AudioAction.Stop]: 'stop',
   [AudioAction.Seek]: 'seek',
   [AudioAction.Next]: 'next',
   [AudioAction.Prev]: 'prev',
   [AudioAction.JumpTo]: 'jump_to',
};

async function sendAction<A extends AudioAction>(
   action: A,
   args: Record<string, unknown>
): Promise<AudioActionResponse<A>> {
   const command = actionCommands[action];

   const response = await invoke<AudioActionResponse<A>>(`plugin:audio|${command}`, args);

   response.player = attachPlayer(response.player);
   return response;
}

async function sendSetting(
   command: string,
   args: Record<string, unknown>
): Promise<PlayerWithAnyStatus> {
   const state = await invoke<PlayerState<PlaybackStatus>>(`plugin:audio|${command}`, args);

   return attachPlayer(state);
}

const transportActions = {
   async load(playlist: PlaylistItem[], startIndex?: number) {
      return sendAction(AudioAction.Load, { playlist, startIndex });
   },

   async play() {
      return sendAction(AudioAction.Play, {});
   },

   async pause() {
      return sendAction(AudioAction.Pause, {});
   },

   async stop() {
      return sendAction(AudioAction.Stop, {});
   },

   async seek(position: number) {
      return sendAction(AudioAction.Seek, { position });
   },

   async next() {
      return sendAction(AudioAction.Next, {});
   },

   async prev() {
      return sendAction(AudioAction.Prev, {});
   },

   async jumpTo(index: number) {
      return sendAction(AudioAction.JumpTo, { index });
   },
} satisfies AllAudioActions;

const playerControls = {
   onStateChanged(listener: (change: StateChange) => void): Promise<UnlistenFn> {
      return stateEventManager.addListener(listener);
   },

   onTrackChanged(listener: (change: TrackChange) => void): Promise<UnlistenFn> {
      return trackEventManager.addListener(listener);
   },

   onSettingsChanged(listener: (change: SettingsChange) => void): Promise<UnlistenFn> {
      return settingsEventManager.addListener(listener);
   },

   onTimeUpdate(listener: (time: TimeUpdate) => void): Promise<UnlistenFn> {
      return timeUpdateEventManager.addListener(listener);
   },

   async setVolume(level: number) {
      return sendSetting('set_volume', { level });
   },

   async setMuted(muted: boolean) {
      return sendSetting('set_muted', { muted });
   },

   async setPlaybackRate(rate: number) {
      return sendSetting('set_playback_rate', { rate });
   },

   async setLoopMode(mode: LoopMode) {
      return sendSetting('set_loop_mode', { mode });
   },
} satisfies PlayerControls;

/**
 * Attaches transport actions (gated by status) and player controls (always available)
 * to a raw {@link PlayerState}, producing a {@link Player} object.
 */
export function attachPlayer<S extends PlaybackStatus>(state: PlayerState<S>): Player<S> {
   const player = { ...state } satisfies PlayerState<S>;

   const actionsForStatus = allowedActions[state.status];

   for (const actionName of actionsForStatus) {
      Object.defineProperty(player, actionName, {
         value: transportActions[actionName],
      });
   }

   for (const [ name, fn ] of Object.entries(playerControls)) {
      Object.defineProperty(player, name, {
         value: fn,
      });
   }

   return player as unknown as Player<S>;
}
