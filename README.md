# Tauri Plugin Audio

[![CI][ci-badge]][ci-url]

Headless, state-driven audio playback API for Tauri 2.x apps with native
transport control integration.

This plugin provides a cross-platform audio playback interface with playlist
support, transport controls (play, pause, stop, seek, next, prev),
volume/rate settings, and OS media integration (lock screen, notification
shade, headphone controls). It is designed to be wrapped by a consuming
app's own API layer.

[ci-badge]: https://github.com/silvermine/tauri-plugin-audio/actions/workflows/ci.yml/badge.svg
[ci-url]: https://github.com/silvermine/tauri-plugin-audio/actions/workflows/ci.yml

## Features

   * State-machine-driven playback with type-safe action gating
   * Playlist playback with `next` / `prev` navigation and auto-advance
   * Loop modes: `off`, `one` (repeat current item), `all` (wrap playlist)
   * OS transport control integration via metadata (title, artist, artwork)
   * Real-time state change events (status, time, volume, etc.)
   * Volume, mute, and playback rate controls
   * Cross-platform support

| Platform | Supported |
| -------- | --------- |
| macOS    | Yes |
| Windows  | Yes |
| Linux    | Yes |
| iOS      | Yes (native planned) |
| Android  | Yes (native planned) |

Playback is provided by the
[Rodio](https://github.com/RustAudio/rodio) audio library via the
standalone `audio-player` crate, which works across all platforms. Full
native support for iOS and Android is planned.

### Platform Notes

#### Android

Android playback uses Rodio's
[Oboe](https://github.com/google/oboe) backend via cpal. The
`audio-player` crate includes a build script that links `libc++_shared`
on Android targets — Tauri's build system automatically bundles the
shared library from the NDK into the APK.

No additional Gradle or manifest configuration is required beyond
Tauri's standard Android setup.

#### iOS

iOS playback uses Rodio's
[CoreAudio](https://developer.apple.com/documentation/coreaudio)
backend via cpal. No additional configuration is required beyond
Tauri's standard iOS setup.

## Getting Started

### Installation

1. Install NPM dependencies:

   ```bash
   npm install
   ```

2. Build the TypeScript bindings:

   ```bash
   npm run build
   ```

3. Build the Rust plugin:

   ```bash
   cargo build
   ```

### Tests

Run all tests (TypeScript and Rust):

```bash
npm test
```

Run TypeScript tests only:

```bash
npm run test:ts
```

Run Rust tests only:

```bash
cargo test --workspace --lib
```

## Install

_This plugin requires a Rust version of at least **1.89**_

### Rust

Add the plugin to your `Cargo.toml`:

`src-tauri/Cargo.toml`

```toml
[dependencies]
tauri-plugin-audio = { git = "https://github.com/silvermine/tauri-plugin-audio" }
```

### JavaScript/TypeScript

Install the JavaScript bindings:

```sh
npm install @silvermine/tauri-plugin-audio
```

## Usage

### Prerequisites

Initialize the plugin in your `tauri::Builder`:

```rust
fn main() {
   tauri::Builder::default()
      .plugin(tauri_plugin_audio::init())
      .run(tauri::generate_context!())
      .expect("error while running tauri application");
}
```

### API

#### Get the player

```ts
import { getPlayer, PlaybackStatus } from '@silvermine/tauri-plugin-audio';

async function checkPlayer() {
   const player = await getPlayer();

   console.debug(`Status: ${player.status}, Time: ${player.currentTime}`);
}
```

#### Load a playlist, play, pause, stop, or seek

The API uses discriminated unions with type guards for compile-time safety.
Only valid transport actions are available based on the player's status.

`load` accepts a playlist (one or more items) and an optional zero-based
`startIndex`. Single-track callers pass a one-item array.

```ts
import {
   getPlayer, PlaybackStatus, hasAction, AudioAction,
} from '@silvermine/tauri-plugin-audio';

async function loadAndPlay() {
   const player = await getPlayer();

   if (player.status === PlaybackStatus.Idle) {
      const { player: ready } = await player.load([
         {
            src: 'https://example.com/song-1.mp3',
            metadata: {
               title: 'Song 1',
               artist: 'Artist',
               artwork: 'https://example.com/cover-1.jpg',
            },
         },
         {
            src: 'https://example.com/song-2.mp3',
            metadata: {
               title: 'Song 2',
               artist: 'Artist',
               artwork: 'https://example.com/cover-2.jpg',
            },
         },
      ]);

      await ready.play();
   }
}

async function managePlayback() {
   const player = await getPlayer();

   if (hasAction(player, AudioAction.Pause)) {
      await player.pause();
   } else if (hasAction(player, AudioAction.Play)) {
      await player.play();
   }
}
```

#### Navigate between playlist items

`next` advances to the next item (with wrap-around when `loopMode` is
`all`). `prev` either restarts the current item (if `currentTime > 3s`) or
moves to the previous item, mirroring the iOS lock-screen convention.

```ts
import { getPlayer, hasAction, AudioAction } from '@silvermine/tauri-plugin-audio';

async function skipForward() {
   const player = await getPlayer();

   if (hasAction(player, AudioAction.Next)) {
      await player.next();
   }
}

async function skipBack() {
   const player = await getPlayer();

   if (hasAction(player, AudioAction.Prev)) {
      await player.prev();
   }
}
```

#### Adjust settings

Volume, mute, playback rate, and loop-mode controls are always available
regardless of playback status.

```ts
import { getPlayer, LoopMode } from '@silvermine/tauri-plugin-audio';

async function adjustSettings() {
   const player = await getPlayer();

   await player.setVolume(0.5);
   await player.setMuted(false);
   await player.setPlaybackRate(1.5);
   await player.setLoopMode(LoopMode.All);
}
```

#### Listen for state-machine transitions

`onStateChanged` fires when the player's `status` or `error` field
changes. The payload carries only those two fields — playlist /
settings / time updates have their own channels.

```ts
import { getPlayer, PlaybackStatus } from '@silvermine/tauri-plugin-audio';

async function watchPlayback() {
   const player = await getPlayer();

   const unlisten = await player.onStateChanged((change) => {
      console.debug(`Status: ${change.status}`);

      if (change.status === PlaybackStatus.Ended) {
         console.debug('Playback finished');
      }
   });

   // To stop listening:
   unlisten();
}
```

#### Listen for active-track changes

`onTrackChanged` fires after each item finishes loading (initial load,
navigation, or auto-advance) and carries the active `PlaylistItem`
with its merged ID3 metadata (title / artist / artwork).

```ts
import { getPlayer } from '@silvermine/tauri-plugin-audio';

async function watchTrack() {
   const player = await getPlayer();

   const unlisten = await player.onTrackChanged((change) => {
      console.debug(
         `Now playing index ${change.currentIndex}: `
         + `${change.item.metadata?.title ?? change.item.src}`,
      );
   });

   // To stop listening:
   unlisten();
}
```

#### Listen for settings changes

`onSettingsChanged` fires when `volume` / `muted` / `playbackRate` /
`loopMode` is mutated. Only the changed field is set on the payload.

```ts
import { getPlayer } from '@silvermine/tauri-plugin-audio';

async function watchSettings() {
   const player = await getPlayer();

   const unlisten = await player.onSettingsChanged((change) => {
      if (change.volume !== undefined) {
         console.debug(`Volume: ${change.volume}`);
      }
      if (change.muted !== undefined) {
         console.debug(`Muted: ${change.muted}`);
      }
   });

   // To stop listening:
   unlisten();
}
```

#### Listen for time updates

`onTimeUpdate` receives lightweight position updates (`currentTime`,
`duration`). Fires on the playback monitor's tick (~250 ms during
playback) and on user-initiated `seek`, so consumers track position
from a single channel regardless of source.

```ts
import { getPlayer } from '@silvermine/tauri-plugin-audio';

async function trackProgress() {
   const player = await getPlayer();

   const unlisten = await player.onTimeUpdate((time) => {
      const pct = time.duration > 0
         ? (time.currentTime / time.duration) * 100
         : 0;

      console.debug(`${time.currentTime}s / ${time.duration}s (${pct.toFixed(1)}%)`);
   });

   // To stop listening:
   unlisten();
}
```

### State Machine

The player follows a state machine where transport actions are gated by
the current `PlaybackStatus`:

| Status    | Allowed Actions                              |
| --------- | -------------------------------------------- |
| Idle      | load                                         |
| Loading   | stop                                         |
| Ready     | play, seek, stop, next, prev, jumpTo         |
| Playing   | pause, seek, stop, next, prev, jumpTo        |
| Paused    | play, seek, stop, next, prev, jumpTo         |
| Ended     | play, seek, load, stop, next, prev, jumpTo   |
| Error     | load, stop, next, prev, jumpTo               |

Settings (`setVolume`, `setMuted`, `setPlaybackRate`, `setLoopMode`)
and the event subscriptions (`onStateChanged`, `onTrackChanged`,
`onSettingsChanged`, `onTimeUpdate`) are always available regardless
of status.

### Loop modes

`setLoopMode(mode)` accepts:

   * `LoopMode.Off` — emit `Ended` after the last playlist item.
   * `LoopMode.One` — repeat the current item indefinitely.
   * `LoopMode.All` — wrap from the last item back to the first.

## Development Standards

This project follows the
[Silvermine standardization](https://github.com/silvermine/standardization)
guidelines. Key standards include:

   * **EditorConfig**: Consistent editor settings across the team
   * **Markdownlint**: Markdown linting for documentation
   * **Commitlint**: Conventional commit message format
   * **Code Style**: 3-space indentation, LF line endings

### Running Standards Checks

```bash
npm run standards
```

## License

MIT

## Contributing

Contributions are welcome! Please follow the established coding standards
and commit message conventions.
