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
 * import { getPlayer, PlaybackStatus, hasAction, AudioAction }
 *    from '@silvermine/tauri-plugin-audio';
 *
 * const player = await getPlayer();
 *
 * if (player.status === PlaybackStatus.Idle) {
 *    const { player: ready } = await player.load('https://example.com/song.mp3', {
 *       title: 'My Song',
 *       artist: 'Artist Name',
 *    });
 *    const { player: playing } = await ready.play();
 *    console.log('Now playing:', playing.title);
 * }
 *
 * // Listen for state changes (e.g. time progression):
 * const unlisten = await player.listen((updated) => {
 *    console.log('Time:', updated.currentTime, '/', updated.duration);
 * });
 * ```
 */
export async function getPlayer(): Promise<PlayerWithAnyStatus> {
   const state = await invoke<PlayerState<PlaybackStatus>>('plugin:audio|get_state');

   return attachPlayer(state);
}

export * from './types';
