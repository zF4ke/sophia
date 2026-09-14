---
title: Choose a model
sidebar:
  order: 2
---

Open `/settings` and select a configured model profile. Profiles live in `resources/models/model-profiles.json`. The default for a fresh installation is Muse Spark 1.3 Free through OpenCode Zen. Existing installations retain their saved selection until an operator changes it.

## OpenCode Zen

The `museSpark13Zen` profile uses `muse-spark-1.3-contributor-free` at `https://opencode.ai/zen/v1`. Set `OPENCODE_API_KEY` in `.env`. A free model still requires authenticated access in this setup. Free availability is provider-controlled.

This model uses the Responses API. Sophia converts tool calls and tool results into Responses items and sends its task ID as session metadata. Sophia owns the agent loop, permissions, memory and tools. It does not run the OpenCode agent. The client identifies itself as Sophia.

The profile explicitly selects `api: "responses"` and `reasoningEffort: "high"`. Supported inputs in this adapter are text and images. Video analysis uses sampled images; audio transcription needs another compatible profile.

OpenCode describes the Contributor model as permitting training on prompts and completions. Check the [provider's current availability and data policy](https://opencode.ai/docs/zen/) before sending private material.

## OpenRouter

GLM 5.3 Flash, GPT-OSS 120B, Ling 3.0 Flash and Muse Spark 1.3 Contributor remain selectable. These use your OpenRouter key. The Contributor OpenRouter profile and the free Zen profile are distinct. Check model prices and modalities before changing them; configured estimates are not a live price feed.

## Local models

Start a compatible local server, such as LM Studio. Copy its exact model ID into the `localLmStudio` profile and match `contextWindow` to the loaded configuration. Its default URL is `http://127.0.0.1:1234/v1` on the machine running Sophia.

A profile's `baseUrl` selects its endpoint. Set `api` explicitly if it uses Responses. Set `apiKeyEnv` only when the endpoint requires authentication. Do not embed credentials in URLs. A local profile does not silently fall back to a remote model.

## Roles and capacity

The selected profile handles ordinary model work. Compaction can use its separately configured summarizer profile. Modalities restrict what a profile can receive. Concurrency controls active requests; context capacity controls the next request size. Neither sets a total task spending budget.

Use [cost reports](../../use/costs/) to inspect retained attempts, including background work and retries.
