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

   /** No playlist is loaded. */
   Idle = 'idle',

   /** A playlist item is being fetched or decoded. */
   Loading = 'loading',

   /** Current item is loaded and ready to play. */
   Ready = 'ready',

   /** Audio is currently playing. */
   Playing = 'playing',

   /** Audio playback is paused. */
   Paused = 'paused',

   /** Reached the end of the playlist with looping disabled. */
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
   Load = 'load',
   Play = 'play',
   Pause = 'pause',
   Stop = 'stop',
   Seek = 'seek',
   Next = 'next',
   Prev = 'prev',
   JumpTo = 'jumpTo',
}

/**
 * How the player advances when the current item finishes.
 *
 * - `Off` — stop after the last item; emit {@link PlaybackStatus.Ended}.
 * - `One` — repeat the current item indefinitely.
 * - `All` — wrap from the last item back to the first.
 */
export enum LoopMode {
   Off = 'off',
   One = 'one',
   All = 'all',
}

/**
 * Metadata for an audio source, used for OS transport control integration
 * (lock screen, notification shade, headphone controls, etc.).
 */
export interface AudioMetadata {
   title?: string;
   artist?: string;
   artwork?: string;
}

/**
 * A single item in a playlist.
 */
export interface PlaylistItem {

   /** URL or file path of the audio source. */
   src: string;

   /** Optional metadata for OS transport controls. */
   metadata?: AudioMetadata;
}

/**
 * The complete state of the audio player at a point in time.
 *
 * `currentTime` and `duration` refer to the active playlist item, identified
 * by `currentIndex` into `playlist`.
 */
export interface PlayerState<S extends PlaybackStatus> {
   status: S;
   playlist: PlaylistItem[];
   currentIndex: number | null;
   currentTime: number;
   duration: number;
   volume: number;
   muted: boolean;
   playbackRate: number;
   loopMode: LoopMode;
   error: string | null;
}

/**
 * Response from a transport action (load, play, pause, stop, seek, next, prev).
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
    * Load a playlist of audio sources.
    *
    * @param playlist - One or more {@link PlaylistItem}s. Empty playlists are rejected.
    * @param startIndex - Zero-based index into the playlist to start from. Defaults to 0.
    * @returns The action response with the updated player state.
    */
   [AudioAction.Load]: (
      playlist: PlaylistItem[],
      startIndex?: number,
   ) => Promise<AudioActionResponse<AudioAction.Load>>;

   /** Start or resume playback. */
   [AudioAction.Play]: () => Promise<AudioActionResponse<AudioAction.Play>>;

   /** Pause playback. */
   [AudioAction.Pause]: () => Promise<AudioActionResponse<AudioAction.Pause>>;

   /** Stop playback and unload the playlist, resetting to {@link PlaybackStatus.Idle}. */
   [AudioAction.Stop]: () => Promise<AudioActionResponse<AudioAction.Stop>>;

   /**
    * Seek to a position in the active item.
    *
    * @param position - The time in seconds to seek to.
    */
   [AudioAction.Seek]: (position: number) => Promise<AudioActionResponse<AudioAction.Seek>>;

   /**
    * Advance to the next playlist item, with wrap-around when
    * {@link LoopMode.All} is set. Transitions to {@link PlaybackStatus.Ended}
    * at the end of a non-looping playlist.
    */
   [AudioAction.Next]: () => Promise<AudioActionResponse<AudioAction.Next>>;

   /**
    * Move to the previous item, or restart the current item if `currentTime`
    * is greater than 3 seconds. Wraps to the last item when
    * {@link LoopMode.All} is set.
    */
   [AudioAction.Prev]: () => Promise<AudioActionResponse<AudioAction.Prev>>;

   /**
    * Jump directly to a specific item in the playlist by index. Jumping to
    * the currently active index restarts that item from the beginning.
    *
    * @param index - Zero-based index into the loaded playlist.
    */
   [AudioAction.JumpTo]: (index: number) => Promise<AudioActionResponse<AudioAction.JumpTo>>;
}

/**
 * Lightweight time update payload. Emitted by the playback monitor (~250 ms
 * tick) and by user-initiated `seek` so consumers learn about position
 * changes from a single channel regardless of source.
 */
export interface TimeUpdate {
   currentTime: number;
   duration: number;
}

/**
 * Status / error transition payload carried on `state-changed` events.
 *
 * Compact by design — fires only when the state-machine status (or its
 * accompanying error message) changes. Settings, navigation, and time
 * updates have their own channels.
 */
export interface StateChange {
   status: PlaybackStatus;
   error: string | null;
}

/**
 * Active-track payload carried on `track-changed` events.
 *
 * Fires after each item finishes loading (initial load, navigation, or
 * auto-advance) and carries the freshly-enriched {@link PlaylistItem} so
 * consumers receive ID3-merged title / artist / artwork without having
 * to query the playlist.
 */
export interface TrackChange {
   currentIndex: number;
   duration: number;
   item: PlaylistItem;
}

/**
 * Partial settings update carried on `settings-changed` events. Only the
 * field whose value changed is set; absent fields are unchanged.
 */
export interface SettingsChange {
   volume?: number;
   muted?: boolean;
   playbackRate?: number;
   loopMode?: LoopMode;
}

/**
 * Settings and subscriptions that are always available regardless of playback status.
 * These do not participate in the state machine gating.
 */
export interface PlayerControls {

   /**
    * Listen for state-machine transitions (status / error changes). To avoid
    * memory leaks, call the `unlisten` function returned by the promise
    * when no longer needed.
    *
    * For navigation between playlist items use {@link onTrackChanged}; for
    * settings mutations use {@link onSettingsChanged}; for high-frequency
    * position updates use {@link onTimeUpdate}.
    */
   onStateChanged: (listener: (change: StateChange) => void) => Promise<UnlistenFn>;

   /**
    * Listen for active-track changes. Fires after each item finishes loading
    * with the enriched {@link PlaylistItem} (title / artist / artwork merged
    * from any embedded ID3 metadata).
    */
   onTrackChanged: (listener: (change: TrackChange) => void) => Promise<UnlistenFn>;

   /**
    * Listen for settings mutations (`volume`, `muted`, `playbackRate`,
    * `loopMode`). Only the changed field is set on the payload.
    */
   onSettingsChanged: (listener: (change: SettingsChange) => void) => Promise<UnlistenFn>;

   /**
    * Listen for playback-position updates (~250 ms during playback, plus
    * one-shot updates from user-initiated `seek`).
    */
   onTimeUpdate: (listener: (time: TimeUpdate) => void) => Promise<UnlistenFn>;

   /**
    * Set the volume level.
    *
    * @param level - Volume level between 0.0 (silent) and 1.0 (maximum).
    */
   setVolume: (level: number) => Promise<PlayerWithAnyStatus>;

   /**
    * Mute or unmute the audio.
    *
    * @param muted - `true` to mute, `false` to unmute.
    */
   setMuted: (muted: boolean) => Promise<PlayerWithAnyStatus>;

   /**
    * Set the playback speed.
    *
    * @param rate - Playback rate where 1.0 is normal speed.
    */
   setPlaybackRate: (rate: number) => Promise<PlayerWithAnyStatus>;

   /**
    * Set the loop behaviour for end-of-track auto-advance.
    *
    * @param mode - One of {@link LoopMode.Off}, {@link LoopMode.One},
    *   or {@link LoopMode.All}.
    */
   setLoopMode: (mode: LoopMode) => Promise<PlayerWithAnyStatus>;
}

// Only these transport actions are allowed for each given PlaybackStatus:
export const allowedActions = {
   [PlaybackStatus.Idle]: [
      AudioAction.Load,
   ],
   [PlaybackStatus.Loading]: [
      AudioAction.Stop,
   ],
   [PlaybackStatus.Ready]: [
      AudioAction.Play,
      AudioAction.Seek,
      AudioAction.Stop,
      AudioAction.Next,
      AudioAction.Prev,
      AudioAction.JumpTo,
   ],
   [PlaybackStatus.Playing]: [
      AudioAction.Pause,
      AudioAction.Seek,
      AudioAction.Stop,
      AudioAction.Next,
      AudioAction.Prev,
      AudioAction.JumpTo,
   ],
   [PlaybackStatus.Paused]: [
      AudioAction.Play,
      AudioAction.Seek,
      AudioAction.Stop,
      AudioAction.Next,
      AudioAction.Prev,
      AudioAction.JumpTo,
   ],
   [PlaybackStatus.Ended]: [
      AudioAction.Play,
      AudioAction.Seek,
      AudioAction.Load,
      AudioAction.Stop,
      AudioAction.Next,
      AudioAction.Prev,
      AudioAction.JumpTo,
   ],
   // From Error, the user can either reload the entire playlist or skip
   // past the broken item via next/prev/jumpTo (the rest of the playlist
   // is preserved). Stop is also allowed so consumers can tear down cleanly.
   [PlaybackStatus.Error]: [
      AudioAction.Load,
      AudioAction.Stop,
      AudioAction.Next,
      AudioAction.Prev,
      AudioAction.JumpTo,
   ],
} as const satisfies Record<PlaybackStatus, AudioAction[] | []>;

export const expectedStatusesForAction = {
   [AudioAction.Load]: [ PlaybackStatus.Ready ],
   [AudioAction.Play]: [ PlaybackStatus.Playing ],
   [AudioAction.Pause]: [ PlaybackStatus.Paused ],
   [AudioAction.Stop]: [ PlaybackStatus.Idle ],
   [AudioAction.Seek]: [
      PlaybackStatus.Ready,
      PlaybackStatus.Playing,
      PlaybackStatus.Paused,
      PlaybackStatus.Ended,
   ],
   // `next` advances to the next item, preserving the prior status (Ready /
   // Playing / Paused), or transitions to Ended when falling off the end of a
   // non-looping playlist.
   [AudioAction.Next]: [
      PlaybackStatus.Ready,
      PlaybackStatus.Playing,
      PlaybackStatus.Paused,
      PlaybackStatus.Ended,
   ],
   // `prev` either restarts the current item (preserves Ready/Playing/Paused,
   // or stays Ended if invoked from Ended after the >3s threshold) or advances
   // to a previous item (preserves Ready/Playing/Paused).
   [AudioAction.Prev]: [
      PlaybackStatus.Ready,
      PlaybackStatus.Playing,
      PlaybackStatus.Paused,
      PlaybackStatus.Ended,
   ],
   // `jumpTo` either restarts the current item (preserves Ready/Playing/Paused,
   // or stays Ended if jumping to the current index from Ended) or jumps to a
   // different item (preserves Ready/Playing/Paused).
   [AudioAction.JumpTo]: [
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
 */
export function hasAnyAction(player: PlayerWithAnyStatus): boolean {
   return allowedActions[player.status].length > 0;
}
