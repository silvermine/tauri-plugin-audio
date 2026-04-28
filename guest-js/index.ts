import { invoke } from '@tauri-apps/api/core';
import { PlayerState, PlaybackStatus, PlayerWithAnyStatus } from './types';
import { attachPlayer } from './actions';
export { attachPlayer };

/**
 * Gets the current audio player state with transport actions and controls attached.
 *
 * This is the primary entry point for interacting with the audio plugin. The returned
 * {@link Player} object has transport actions (play, pause, etc.) gated by the current
 * {@link PlaybackStatus}, plus always-available controls (setVolume, listen, etc.).
 *
 * @returns The current player state with actions attached.
 *
 * @example
 * ```ts
 * import { getPlayer, PlaybackStatus } from '@silvermine/tauri-plugin-audio';
 *
 * const player = await getPlayer();
 *
 * if (player.status === PlaybackStatus.Idle) {
 *    const { player: ready } = await player.load([
 *       {
 *          src: 'https://example.com/song.mp3',
 *          metadata: { title: 'My Song', artist: 'Artist Name' },
 *       },
 *    ]);
 *    await ready.play();
 * }
 *
 * // Listen for active-track changes (e.g. on auto-advance):
 * const unlistenTrack = await player.onTrackChanged((change) => {
 *    console.log('Now playing:', change.item.metadata?.title);
 * });
 *
 * // Listen for playback-position updates (~250ms during playback):
 * const unlistenTime = await player.onTimeUpdate((time) => {
 *    console.log('Time:', time.currentTime, '/', time.duration);
 * });
 * ```
 */
export async function getPlayer(): Promise<PlayerWithAnyStatus> {
   const state = await invoke<PlayerState<PlaybackStatus>>('plugin:audio|get_state');

   return attachPlayer(state);
}

export * from './types';
