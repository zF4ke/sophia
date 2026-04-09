Judge whether the current Discord evidence is sufficient to answer the user's question reliably.

Question:
{{question}}

Message evidence count: {{message_evidence_count}}
Live evidence count: {{live_evidence_count}}

Evidence:
{{evidence}}

Rules:
- Return strict JSON only.
- Set `sufficient` to true only if the evidence directly answers the question or clearly provides the requested factual basis.
- Set `sufficient` to false if the evidence is related but still missing key facts needed to answer reliably.
- For channel/content requests, evidence is sufficient when the retrieved messages themselves contain the requested items, even if retrieval scores are moderate.
- For "all" or exhaustive requests, require evidence that appears complete enough for that scope.
- Do not require clickable citations to consider the evidence sufficient.

Return this shape:
{
  "sufficient": true,
  "reason": "short explanation",
  "missingInformation": null
}
