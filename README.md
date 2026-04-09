# Tauri Plugin Audio

[![CI][ci-badge]][ci-url]

Headless, state-driven audio playback API for Tauri 2.x apps with native
transport control integration.

This plugin provides a cross-platform audio playback interface with transport
controls (play, pause, stop, seek), volume/rate settings, and OS media
integration (lock screen, notification shade, headphone controls). It is
designed to be wrapped by a consuming app's own API layer.

[ci-badge]: https://github.com/silvermine/tauri-plugin-audio/actions/workflows/ci.yml/badge.svg
[ci-url]: https://github.com/silvermine/tauri-plugin-audio/actions/workflows/ci.yml

## Features

   * State-machine-driven playback with type-safe action gating
   * OS transport control integration via metadata (title, artist, artwork)
   * Real-time state change events (status, time, volume, etc.)
   * Volume, mute, playback rate, and loop controls
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

#### Prepare, play, pause, stop, or seek

The API uses discriminated unions with type guards for compile-time safety.
Only valid transport actions are available based on the player's status.

```ts
import {
   getPlayer, PlaybackStatus, hasAction, AudioAction,
} from '@silvermine/tauri-plugin-audio';

async function prepareAndPlay() {
   const player = await getPlayer();

   if (player.status === PlaybackStatus.Idle) {
      const { player: ready } = await player.prepare(
         'https://example.com/song.mp3',
         {
            title: 'My Song',
            artist: 'Artist Name',
            artwork: 'https://example.com/cover.jpg',
         },
      );

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

#### Adjust settings

Volume, mute, playback rate, and loop controls are always available
regardless of playback status.

```ts
import { getPlayer } from '@silvermine/tauri-plugin-audio';

async function adjustSettings() {
   const player = await getPlayer();

   await player.setVolume(0.5);
   await player.setMuted(false);
   await player.setPlaybackRate(1.5);
   await player.setLoop(true);
}
```

#### Listen for state changes

`listen` receives updates for state transitions (status changes,
volume, settings, errors).

```ts
import { getPlayer, PlaybackStatus } from '@silvermine/tauri-plugin-audio';

async function watchPlayback() {
   const player = await getPlayer();

   const unlisten = await player.listen((updated) => {
      console.debug(`Status: ${updated.status}`);

      if (updated.status === PlaybackStatus.Ended) {
         console.debug('Playback finished');
      }
   });

   // To stop listening:
   unlisten();
}
```

#### Listen for time updates

`onTimeUpdate` receives lightweight, high-frequency updates
(~250ms) carrying only `currentTime` and `duration`, avoiding the
overhead of serializing the full player state on every tick.

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

| Status    | Allowed Actions              |
| --------- | ---------------------------- |
| Idle      | prepare                      |
| Loading   | stop                         |
| Ready     | play, seek, stop             |
| Playing   | pause, seek, stop            |
| Paused    | play, seek, stop             |
| Ended     | play, seek, prepare, stop    |
| Error     | prepare                      |

Settings (`setVolume`, `setMuted`, `setPlaybackRate`, `setLoop`),
`listen`, and `onTimeUpdate` are always available regardless of
status.

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
