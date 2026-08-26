# The safety layer — what it is, where its text comes from, and what happens when it misses

*Written 2026-08-09 for the pre-deployment review. Every number here was measured
against the deployed configuration (`configs/chatbot.yaml`, index
`structure-512-0__e5-large__mv`, generator `google/gemma-4-12B-it`) on that date,
not carried over from an earlier draft. Companion to
[`query_phrasing_probe.md`](query_phrasing_probe.md), which covers how the
counselor's phrasing changes what the screen sees; this document covers the layer
itself.*

Two questions this is written to answer directly, because both have been asked and
neither is answerable from the thesis as it stands:

1. **Where does the fixed safety text come from?** (§2)
2. **What happens if the screen misses?** (§3)

---

## 1. The banner and the briefing are two different things

A risk verdict triggers **two independent mechanisms**, and conflating them is the
most common misreading of this system.

| | mechanism | source | can it fail? |
|---|---|---|---|
| **Banner** | fixed string prepended in code | `safety.safe_response` in `configs/chatbot.yaml` | no — it is a string copy |
| **Briefing** | `RISK_NOTE` appended to the user prompt | the model, from retrieved chunks | yes — can miss, drift, or truncate |

The decisive fact:

```bash
grep -c "119" data/derived/cleaned_text.txt          # 0
grep -oci "sejiwa\|healing" data/derived/cleaned_text.txt   # 0
grep -oEc '\b0[0-9]{2,3}-?[0-9]{6,}\b' data/derived/cleaned_text.txt   # 0
```

**The guideline corpus contains no phone numbers at all.** The model therefore
*cannot* produce the crisis line from context, however good retrieval is. The
banner is the only channel that carries it, and any number appearing in a
generated briefing would be a hallucination.

That makes the two failure directions structurally asymmetric:

- **False positive** → one redundant banner plus a mild prompt nudge. The briefing
  is still correct and grounded.
- **False negative** → the crisis number is *absent from the entire output*. Not
  less prominent — absent.

This asymmetry is a property of the architecture, not a weighting preference, and
it is the reason the screen is deliberately over-inclusive.

**Ordering.** In the streaming path (`chatbot.py`, `answer_stream`) the banner is
emitted *before* generation starts, so it survives a timeout, a crash, or the stop
button. Both deployed UIs use that path. The non-streaming `answer()` prepends it
*after* generation instead, so a generation failure there loses it — that path is
reachable only through `chatbot_web.py`'s `/chat`, which is researcher-only.

**Not rated.** `generate_study_answers.py` deliberately excludes the banner from
the frozen briefings, so the psychologists rate model-generated text only. The
consequence — study safety scores measure the generated half and therefore
*understate* deployed safety — is recorded in
`documents/evaluation/protocol_amendment_20260806.md` §1 and must appear in the
limitations.

---

## 2. Where the fixed text comes from

The banner is a **hybrid**, and saying so plainly is the honest answer.

### 2.1 The principle is from the guideline

"Suicide risk requires direct assessment and immediate referral" comes from MI.7
(psychiatric emergency) and the referral system. The banner's closing sentence
cites exactly that, and it is citable.

### 2.2 The contact details are not

They are not in the corpus (§1). They were sourced externally and triangulated,
**re-verified 2026-08-02** (previously 2026-07-13). The audit trail lives in the
`safety:` block of `configs/chatbot.yaml` — read it there, not here, so the two
cannot drift:

- **119 ext. 8** — agreed by all three sources consulted
- **www.healing119.id** — site live; its own page reads *"Hubungi 119 ext. 8 untuk
  darurat bunuh diri"*
- **free and anonymous** — *"disediakan secara gratis … secara anonim"*
- Healing119 is the 2025-07-31 relaunch of SEJIWA (Dit. Keswa + IPK Indonesia)

### 2.3 The ordering is a clinical judgement, with a reason

**IGD / puskesmas comes first, the hotline second.** Healing119 is psychological
first aid, not emergency response: its own site describes it as *"bukan layanan
terapi klinis jangka panjang, melainkan pertolongan pertama psikologis"*, with
sessions capped at 30 minutes. Acute risk needs in-person assessment first; the
hotline is the support layer behind it.

### 2.4 Two deliberate omissions

Both are defensible and should be stated before an examiner finds them.

- **Operating hours are not stated.** Two government sources contradict each
  other — `kesprimkom.kemkes.go.id` (2025-08-07) says *"pagi hingga malam"*,
  `infopublik.id` (2025-12-19) says *"24 jam"* — and the service's own site states
  none. The newer source suggests hours were extended, but this could not be
  confirmed. Refusing to state an unresolved fact to participants is the correct
  call, and it is enforced: `INTERNAL_catatan_item_konselor.md` was the one place
  that still quoted hours, and that is a known correction.
- **112 was dropped deliberately.** It is Indonesia's general emergency dispatch
  line; the template routes to facilities that can perform a psychiatric
  assessment instead. `safe_response` has never named it. Do not add it back.

### 2.5 Where this is weak

- Verified by the researcher, **not by a clinician**, and no supervisor or
  psychologist sign-off is recorded. Getting one, with a name and a date, is cheap
  and closes the objection.
- The audit trail lives in a YAML comment, not in the thesis. The manuscript
  currently says only *"the national Healing119 crisis line (verified against
  Ministry sources)"* — one clause, no date, no sources, and no mention that the
  number is absent from the guideline. §2.1–2.4 above is the material that should
  be in an appendix.
- `configs/chatbot.yaml` instructs: **re-verify the week before the study runs.**

---

## 3. What happens if the screen misses

Five layers, genuinely independent. The question is not whether layer 1 is perfect
— it is not — but whether a miss propagates all the way to the counselor.

### 3.1 Keyword pre-screen

58 terms as of 2026-08-09 (15 → 45 on 2026-08-06 → 58 today). Deliberately
over-inclusive: a hit only raises the banner and adds a prompt instruction.

Measured on the 50 frozen study questions: **12 of the 24 risk cases** are caught
here. Probed against 25 hand-written variants (punctuation, spacing, run-together
typos, informal register): **0 misses, 0 false positives**.

Its known blind spot is euphemism, and that is not fixable by widening the list —
see `query_phrasing_probe.md` §2.2 and §5. Terms like *loncat*, *terjun*, *kereta*
are common in ordinary text and would fire constantly.

### 3.2 LLM classifier

Runs only when the keyword screen misses. It carries **the other 12 of 24** risk
cases, so it is load-bearing, not a formality.

Probed 2026-08-09 with 8 risky inputs deliberately phrased to evade the keyword
list (*"sudah menyiapkan tali di kamarnya"*, *"menulis surat perpisahan"*, *"dunia
lebih baik tanpa dirinya"*): **8/8 caught**, and 2/2 control questions correctly
returned AMAN.

Decoding is greedy (temperature 0.0), so the verdict is stable for a fixed input.
The parse fails **closed**: anything that is not a clean `AMAN` — a negation, a
preamble, an empty completion, a refusal — is treated as risk. A latent inversion
bug here (`"TIDAK AMAN"` parsed as safe) was found and fixed on 2026-08-09; it had
never fired, in 36 recorded verdicts or 10 live probes, but it is now covered by
regression tests.

**It is a hard dependency.** In dry-run the LLM stage is skipped entirely,
silently removing half the risk detection. Since 2026-08-09 both
`serve_chatbot.sh` and `chatbot_app.py` refuse to publish a link when the engine
falls back to dry-run, so this can no longer happen unnoticed — a 2026-08-01
launch did exactly that and the pre-send check reported PASS.

### 3.3 The prompt itself

System-prompt rule 6 requires that *any* mention of suicide or self-harm risk —
including as something merely for the counselor to check — also names the
escalation route, written without `[n]` as a standard emergency step.

Re-measured 2026-08-09 over the 50 frozen briefings: **46 mention risk, 46 give a
route, 0 without.** Before rule 6 (`prompt_fix_20260801.md`) it was 34 briefings
raising risk with no route. So even when the banner does not fire, the briefing
routes.

### 3.4 Retrieval

`risk_suicide_emergency` **recall@5 = 1.000** (n = 17) on the deployment index
`structure-512-0__e5-large__mv` — `outputs/analysis/retrieval_metrics_by_type.csv`.
If the question is about suicide risk, the MI.7 emergency material is in context
regardless of what the screen decided.

### 3.5 The counselor

The tool is decision support. Its output never reaches the help-seeker, and the
counselor has read the original text themselves — they are not relying on the
briefing to tell them a case is serious.

### 3.6 Evidence the layers actually compensate

Use this rather than asserting redundancy in the abstract. The manuscript already
concedes that on **1 of 24** risk-flagged questions the danger-signs section failed
to surface the risk — *and the banner covered it*. That is the reverse failure,
caught by an independent layer.

---

## 4. The honest limit

**The false-negative rate is not established out-of-sample.** The 50-question set
is where the flags were defined, so "24/24 caught" is partly circular: for the
LLM-only subset, the frozen labels were produced by the same classifier being
evaluated. If asked *"what is your screen's sensitivity?"*, the truthful answer
today is that it has not been measured on held-out data.

This is cheap to fix and the material already exists:

- `data/derived/questions_heldout25.jsonl` — a real held-out set, never used for this
- the 323 scraped Alodokter questions outside the frozen sample of 50
- independent adjudication of disputed cases by a different model family
  (Qwen3-32B is already the judge) or by a clinician

Until that is run, report the screen as *"catches every risk case in the frozen
study set, sensitivity not established out-of-sample"* — which is defensible.
Claiming a sensitivity figure from the 50 is not.

---

## 5. Related documents

| | |
|---|---|
| how phrasing changes what the screen sees; euphemism gap | [`query_phrasing_probe.md`](query_phrasing_probe.md) |
| rule 6 and its measured effect | [`prompt_fix_20260801.md`](prompt_fix_20260801.md) |
| why the banner is excluded from rated material | [`evaluation/protocol_amendment_20260806.md`](evaluation/protocol_amendment_20260806.md) |
| the banner text, keyword list, and the audit trail itself | `configs/chatbot.yaml`, `safety:` block |
| confirming the generator is up before a session | [`gradio_link_runbook.md`](gradio_link_runbook.md) §3 |
