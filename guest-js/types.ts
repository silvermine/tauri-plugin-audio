import type { UnlistenFn } from '@tauri-apps/api/event';

/**
 * Represents the current playback status of the audio player.
 *
 * Modeled after common media player states (inspired by Vidstack's player state model),
 * adapted for headless native audio playback with transport control integration.
 *
 * Use the `status` field on a {@link Player} object to determine which transport
 * actions are available. TypeScript will automatically narrow the available methods
 * based on the status.
 *
 * @example
 * ```ts
 * if (player.status === PlaybackStatus.Ready) {
 *    await player.play(); // TypeScript knows play() is available
 * }
 * ```
 */
export enum PlaybackStatus {

   /** No audio source is loaded. */
   Idle = 'idle',

   /**
    * An audio source is being loaded. Reserved for the real implementation
    * where loading is asynchronous. The mock transitions directly from
    * Idle to Ready.
    */
   Loading = 'loading',

   /** Audio source is loaded and ready to play. */
   Ready = 'ready',

   /** Audio is currently playing. */
   Playing = 'playing',

   /** Audio playback is paused. */
   Paused = 'paused',

   /** Audio playback has reached the end. */
   Ended = 'ended',

   /** An error occurred during loading or playback. */
   Error = 'error',
}

/**
 * Transport actions that are gated by the current {@link PlaybackStatus}.
 *
 * Only the actions listed in {@link allowedActions} for a given status will be
 * attached to the {@link Player} object.
 */
export enum AudioAction {
   Prepare = 'prepare',
   Play = 'play',
   Pause = 'pause',
   Stop = 'stop',
   Seek = 'seek',
}

/**
 * Metadata for the audio source, used for OS transport control integration
 * (lock screen, notification shade, headphone controls, etc.).
 */
export interface AudioMetadata {
   title?: string;
   artist?: string;
   artwork?: string;
}

/**
 * The complete state of the audio player at a point in time.
 *
 * Inspired by Vidstack's player state model, this captures all relevant playback
 * properties: source info, timing, volume, and error state.
 */
export interface PlayerState<S extends PlaybackStatus> {
   status: S;
   src: string | null;
   title: string | null;
   artist: string | null;
   artwork: string | null;
   currentTime: number;
   duration: number;
   volume: number;
   muted: boolean;
   playbackRate: number;
   loop: boolean;
   error: string | null;
}

/**
 * Response from a transport action (prepare, play, pause, stop, seek).
 *
 * Wraps the resulting player state with status-expectation metadata so callers
 * can detect unexpected state transitions.
 */
export interface AudioActionResponse<A extends AudioAction = AudioAction> {
   player: PlayerWithAnyStatus;
   expectedStatus: ExpectedStatusesForAction<A>;
   isExpectedStatus: boolean;
}

/**
 * Signatures for all transport actions. Only the subset allowed for a given
 * {@link PlaybackStatus} will be attached to the {@link Player} object.
 */
export interface AllAudioActions {

   /**
    * Prepare an audio source.
    *
    * @param src - URL or file path of the audio source.
    * @param metadata - Optional metadata for OS transport
    *   controls (title, artist, artwork).
    * @returns The action response with the updated player state.
    */
   [AudioAction.Prepare]: (src: string, metadata?: AudioMetadata) => Promise<AudioActionResponse<AudioAction.Prepare>>;

   /** Start or resume playback. */
   [AudioAction.Play]: () => Promise<AudioActionResponse<AudioAction.Play>>;

   /** Pause playback. */
   [AudioAction.Pause]: () => Promise<AudioActionResponse<AudioAction.Pause>>;

   /** Stop playback and unload the audio source, resetting to Idle. */
   [AudioAction.Stop]: () => Promise<AudioActionResponse<AudioAction.Stop>>;

   /**
    * Seek to a position in the audio.
    *
    * @param position - The time in seconds to seek to.
    * @returns The action response with the updated player state.
    */
   [AudioAction.Seek]: (position: number) => Promise<AudioActionResponse<AudioAction.Seek>>;
}

/**
 * Lightweight time update payload emitted at high frequency during
 * playback (typically every 250ms). Separated from full state changes
 * to minimize serialization overhead.
 */
export interface TimeUpdate {
   currentTime: number;
   duration: number;
}

/**
 * Settings and subscriptions that are always available regardless of playback status.
 * These do not participate in the state machine gating.
 */
export interface PlayerControls {

   /**
    * Listen for changes to the player state. To avoid memory leaks,
    * call the `unlisten` function returned by the promise when no
    * longer needed.
    *
    * Receives updates for state transitions (status changes, volume,
    * settings, errors). For high-frequency time progression, use
    * {@link onTimeUpdate} instead.
    *
    * @param listener - Callback invoked when the player state
    *   changes.
    * @returns A promise with a function to remove the listener.
    *
    * @example
    * ```ts
    * const unlisten = await player.listen((updated) => {
    *   console.log('Status:', updated.status);
    * });
    *
    * // To stop listening:
    * unlisten();
    * ```
    */
   listen: (listener: (player: PlayerWithAnyStatus) => void) => Promise<UnlistenFn>;

   /**
    * Listen for high-frequency time progression updates during
    * playback (typically every 250ms).
    *
    * This is a lightweight event carrying only `currentTime` and
    * `duration`, avoiding the overhead of serializing the full
    * player state on every tick.
    *
    * @param listener - Callback invoked on each time update.
    * @returns A promise with a function to remove the listener.
    *
    * @example
    * ```ts
    * const unlisten = await player.onTimeUpdate((time) => {
    *   progressBar.value = time.currentTime / time.duration;
    * });
    * ```
    */
   onTimeUpdate: (listener: (time: TimeUpdate) => void) => Promise<UnlistenFn>;

   /**
    * Set the volume level.
    *
    * @param level - Volume level between 0.0 (silent) and 1.0 (maximum).
    * @returns The updated player state with actions attached.
    */
   setVolume: (level: number) => Promise<PlayerWithAnyStatus>;

   /**
    * Mute or unmute the audio.
    *
    * @param muted - `true` to mute, `false` to unmute.
    * @returns The updated player state with actions attached.
    */
   setMuted: (muted: boolean) => Promise<PlayerWithAnyStatus>;

   /**
    * Set the playback speed.
    *
    * @param rate - Playback rate where 1.0 is normal speed.
    * @returns The updated player state with actions attached.
    */
   setPlaybackRate: (rate: number) => Promise<PlayerWithAnyStatus>;

   /**
    * Enable or disable looping.
    *
    * @param loop - `true` to loop, `false` for single playback.
    * @returns The updated player state with actions attached.
    */
   setLoop: (loop: boolean) => Promise<PlayerWithAnyStatus>;
}

// Only these transport actions are allowed for each given PlaybackStatus:
export const allowedActions = {
   [PlaybackStatus.Idle]: [
      AudioAction.Prepare,
   ],
   [PlaybackStatus.Loading]: [
      AudioAction.Stop,
   ],
   [PlaybackStatus.Ready]: [
      AudioAction.Play,
      AudioAction.Seek,
      AudioAction.Stop,
   ],
   [PlaybackStatus.Playing]: [
      AudioAction.Pause,
      AudioAction.Seek,
      AudioAction.Stop,
   ],
   [PlaybackStatus.Paused]: [
      AudioAction.Play,
      AudioAction.Seek,
      AudioAction.Stop,
   ],
   [PlaybackStatus.Ended]: [
      AudioAction.Play,
      AudioAction.Seek,
      AudioAction.Prepare,
      AudioAction.Stop,
   ],
   [PlaybackStatus.Error]: [
      AudioAction.Prepare,
   ],
} as const satisfies Record<PlaybackStatus, AudioAction[] | []>;

export const expectedStatusesForAction = {
   [AudioAction.Prepare]: [ PlaybackStatus.Ready ],
   [AudioAction.Play]: [ PlaybackStatus.Playing ],
   [AudioAction.Pause]: [ PlaybackStatus.Paused ],
   [AudioAction.Stop]: [ PlaybackStatus.Idle ],
   [AudioAction.Seek]: [
      PlaybackStatus.Ready,
      PlaybackStatus.Playing,
      PlaybackStatus.Paused,
      PlaybackStatus.Ended,
   ],
} as const satisfies Record<AudioAction, PlaybackStatus[]>;

type ActionsFns<S extends PlaybackStatus> = Pick<AllAudioActions, typeof allowedActions[S][number]>;
type AllowedActionsForStatus<S extends PlaybackStatus> = ActionsFns<S> extends never ? object : ActionsFns<S>;

/**
 * A player in a specific status, with only the transport actions valid for that status
 * plus always-available {@link PlayerControls} (listen, setVolume, etc.).
 */
export type Player<S extends PlaybackStatus> = PlayerState<S> & AllowedActionsForStatus<S> & PlayerControls;

/**
 * Union type representing a player in any status.
 *
 * To narrow to a specific status, use either {@link hasAction} or the `status`
 * field as a discriminator.
 *
 * @example
 * ```ts
 * if (hasAction(player, AudioAction.Play)) {
 *    await player.play();
 * }
 *
 * // Or:
 * if (player.status === PlaybackStatus.Paused) {
 *    await player.play(); // TypeScript knows play() is available
 * }
 * ```
 */
export type PlayerWithAnyStatus = { [T in PlaybackStatus]: Player<T> }[PlaybackStatus];

export type ExpectedStatusesForAction<A extends AudioAction> = (typeof expectedStatusesForAction)[A][number];
export type UnexpectedStatusesForAction<A extends AudioAction> = Exclude<PlaybackStatus, ExpectedStatusesForAction<A>>;

export type ExpectedStatesForAction<A extends AudioAction> = Extract<PlayerWithAnyStatus, Pick<AllAudioActions, A>>;
export type UnexpectedStatesForAction<A extends AudioAction> = Exclude<PlayerWithAnyStatus, ExpectedStatesForAction<A>>;

/**
 * Type guard that checks whether a transport action is available on the given player.
 *
 * @example
 * ```ts
 * if (hasAction(player, AudioAction.Pause)) {
 *    await player.pause(); // TypeScript narrows the type
 * }
 * ```
 */
export function hasAction<A extends AudioAction>(
   player: PlayerWithAnyStatus,
   actionName: A
): player is Extract<PlayerWithAnyStatus, Pick<AllAudioActions, A>> {
   return (allowedActions[player.status] as AudioAction[]).includes(actionName);
}

/**
 * Checks whether the player has any transport actions available.
 *
 * Currently all statuses have at least one transport action, so this
 * always returns `true`. It is provided for forward-compatibility if
 * terminal statuses (with no actions) are added in the future.
 */
export function hasAnyAction(player: PlayerWithAnyStatus): boolean {
   return allowedActions[player.status].length > 0;
}
