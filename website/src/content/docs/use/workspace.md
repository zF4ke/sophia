---
title: Files, code, images and media
sidebar:
  order: 6
---

Attach a file and describe the result you want. Sophia can import supported attachments into a task workspace, inspect its files, execute Python, JavaScript or shell code, and retain outputs for later revision.

## Analyze and create



> **You**: [attendance.csv] Plot attendance by event and give me a CSV with duplicate registrations removed.
>
> **Sophia**: Two rows repeat the same registration ID. I kept the first occurrence. Here are the chart and cleaned CSV. [Files]
>
> **You**: Use the latest registration instead, then regenerate both.

Code runs in a disposable Docker container. The task's returned files persist separately. The container has no host credentials, network, writable host mounts or elevated privileges. Transfers and individual operations have limits; those limits do not cap the whole task. If Docker is unavailable, Sophia reports that failure instead of running code on your PC.

Files remain owned and source-scoped. Publishing to Discord is a separate authorized action. Inspect and export files through `/tasks`; add `ephemeral:true` when you want a private download.

## Images and video

A vision-capable profile can inspect attached images. Video inspection samples timestamped frames; it is not proof that every frame was watched. Ask for the relevant segment if timing matters. Expired media links and unsupported formats may require a fresh attachment.



> **You**: [clip.mp4] Where does the overlay disappear?
>
> **Sophia**: It's visible in the sampled frame at 00:04 and absent at 00:06. I need denser sampling between those points to identify the transition more precisely.

## Audio

Audio transcription requires a compatible configured audio model. A text-only or image-only model does not gain audio support from the sandbox. The current Responses adapter supports text and images; select a compatible chat-completions audio profile for transcription.

Sophia can also calculate expressions and measure text length without a full code run. Use these for exact arithmetic or a character limit. See [workspace architecture](../../build/workspace-and-media/).
