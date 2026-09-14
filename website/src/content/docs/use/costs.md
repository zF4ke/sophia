---
title: Track costs and usage
sidebar:
  order: 9
---

Open `/costs` for your own usage across locations. Operators can open `/settings` → **Custos** to see the whole installation, including unowned maintenance work. `/costs` posts your report in the channel by default. Use `ephemeral:true` for a private report. Installation-wide usage uses the settings panel’s visibility. Controls belong to the person who opened the report.

Choose 24 hours, 7 days, 30 days, or the whole retained record. The report shows known estimated USD cost, attempts, failed attempts, background work, reported input/output tokens, missing usage data and up to eight models ordered by known cost.

## Read the numbers correctly

<div class="usage-example" aria-label="Simulated cost report">
  <div class="usage-heading"><strong>Usage report</strong><span>Simulated report for the last 7 days</span></div>
  <div class="usage-total"><strong>$0.0240</strong><span>USD known estimated cost</span></div>
  <dl class="usage-metrics"><div><dt>Attempts</dt><dd>12</dd></div><div><dt>Failed</dt><dd>1</dd></div><div><dt>Background</dt><dd>2</dd></div></dl>
  <p class="usage-note">1 attempt has no calculable cost.</p>
</div>
This does not mean the unknown attempt was free. A provider error may omit usage. Retries are separate attempts. Prices are captured when an attempt is recorded, so editing today's model price does not rewrite past estimates.

A zero-price model can have a known zero estimate when complete usage is returned. Free availability is controlled by the provider. Estimates do not account for every possible billing adjustment, cache discount or provider service charge. Your provider's invoice remains authoritative.

## Accounting does not stop work

This panel adds no spending cap. Context occupancy and cumulative token usage are different quantities. Forgetting a task removes its associated usage records from these totals, so this is a report of retained records rather than a permanent financial ledger.

For one task's calls and evidence, open `/tasks task_id:...`.
