# Workspace and media

Sophia stores files in the owning task's durable ledger. Another actor or channel cannot read that workspace. File changes on the same task run serially, and invalid container exports leave the previous files intact.

`sandbox_import` accepts current-turn or historical Discord attachment IDs. Historical imports use the original message link or the local attachment index to fetch a fresh URL. It downloads only from Discord's HTTPS media hosts, rejects redirects, validates the destination path, and stops on cancellation or an oversized transfer. It does not fetch arbitrary URLs.

`sandbox_run` starts a fresh Docker container with the task's files supplied through standard input. There are no host mounts, Docker socket mounts, network access, or inherited proxy values. The container runs as an unprivileged user with a read-only root, dropped capabilities, process and memory limits, and temporary writable directories. Python includes pandas, matplotlib, Pillow, pypdf and openpyxl. Node and ffmpeg are available. Files written under `/workspace` return to the task after validation.

Each execution permits at most 300 seconds. A transfer permits 100 files and 20 MiB. These are operation limits; they do not cap a task's calls or total duration. Split large work into parts. An unavailable Docker engine or image returns an error; generated code never falls back to the host.

Build the local image with `npm run sandbox:build` after starting Docker's Linux engine. Configure the image name and availability in `settings.sandbox`. The current container flags follow the [Docker run reference](https://docs.docker.com/reference/cli/docker/container/run/).

`sandbox_inspect` supplies up to eight image previews, or samples one video at up to eight requested timestamps. Media decoders run in the same isolated backend. The following model turn receives actual image content, labelled with the task filename and sample timestamp. A sampled frame does not establish what occurred between samples or what the audio contains. Use `ffprobe` in the workspace to inspect video duration before sampling.

Message attachments and `/talk attachment` can also supply images directly. Supported Discord image URLs become chat image parts or Responses image inputs. A declared text-only model receives metadata only. These formats follow the [image input documentation](https://developers.openai.com/api/docs/guides/images-vision). File names and media content remain source material, not instructions.

`sandbox_publish` sends an existing output through the normal write approval policy and returns the message ID and link. Delivery supports guild channels and the authenticated owner's DM, with files up to 8 MiB. Visual inspection requires declared image support before executing the decoder. Expired signed attachment URLs stay metadata-only.

`sandbox_transcribe` extracts a mono WAV clip with ffmpeg inside the container and sends it to a configured chat-completions profile declaring audio input. Select up to 180 seconds per call; process longer recordings in consecutive windows. Audio bytes use OpenRouter's documented [`input_audio` contract](https://openrouter.ai/docs/guides/overview/multimodal/audio). The transcript is saved with its source file, requested time range and model. Inaudible speech stays marked; word timestamps and speaker identities are not inferred. Binary media are omitted from model trace logs. Configured OpenRouter profile modalities were refreshed from the public model catalog on 14 September 2026; local profiles require operator declarations.

Card scripts use this same container backend. The inner JavaScript VM supplies card helpers; it is not the host isolation boundary. The host validates returned state and messages. External sends still require the clicking user's authority and normal approval.

Deterministic tests cover isolation arguments, path rejection, file ownership, concurrent updates, malformed exports and source-labelled previews. The Linux Docker engine and image build are now verified on this machine. Real execution, isolation and card-script checks passed; see [live acceptance](live-acceptance.md) for exact scenarios and remaining limits.

## Historical Discord attachments

`sandbox_import` accepts `message_url` alongside `attachment_id` and `path`. It checks source-channel access, fetches the original message through DiscordHistoryReader to renew its signed attachment URL, and preserves message/channel provenance on the downloaded file. When omitted, the original message can be found by attachment ID in the local index. Use `sandbox_inspect` to see imported images/video. CDN URL expiry is not evidence of deletion. A missing original message, removed attachment or permission failure is reported separately. The documented Discord message API renews signed URLs; no undocumented URL-refresh endpoint is required.
