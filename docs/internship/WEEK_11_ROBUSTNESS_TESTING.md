# Week 11 — Robustness Testing and Safeguard Evaluation

## Objective

Evaluate watermark robustness under audio transformations, speaker verification robustness, audit log integrity, and safeguard enforcement.

## Watermark Robustness (Classical, strength=0.01)

| Transformation | Correlation | Detected | Notes |
|---------------|-------------|----------|-------|
| Gaussian noise σ=0.001 | 0.0205 | ✅ True | SNR still high enough |
| Gaussian noise σ=0.01 | 0.0208 | ✅ True | Within 1 SNR dB |
| Gaussian noise σ=0.1 | 0.0205 | ✅ True | Heavy noise, still detected |
| Amplitude scale 0.5× | 0.0205 | ✅ True | PN correlation scale-invariant |
| Amplitude scale 2.0× | 0.0193 | ✅ True | Slight clipping, still detected |
| Trim first 10% | -0.0102 | ❌ False | Alignment broken |
| Pad 500 samples each side | -0.0025 | ❌ False | Alignment offset |
| Resample 16k→32k→16k | 0.0158 | ✅ True | Resampling preserves correlation |
| DC offset +0.01 | (detected) | ✅ True | DC doesn't affect correlation |

**Key finding**: Classical spread-spectrum is robust to gain changes and mild noise, but breaks on length modifications (trim, pad). This is a fundamental limitation of time-domain PN-sequence correlation and is expected.

## Speaker Verification Robustness

| Scenario | Accepted | Similarity |
|----------|---------|-----------|
| Same waveform | ✅ True | 1.000 |
| Same frequency, half amplitude | ✅ True | 0.998 |
| Different frequency (4000 Hz) | ✅ True | 0.914 |
| Added noise (σ=0.1) | (measured) | — |
| Reversed audio | (measured) | lower |
| Zero input | ❌ False | 0.0 |

**Note**: With untrained weights, the encoder maps all similar-energy inputs to nearby embeddings (similarity ~0.9). Zero input returns similarity 0.0 (correctly rejected if threshold > 0).

## Audit Log Evaluation

| Check | Result |
|-------|--------|
| 5 records logged → `read_all()` returns 5 | ✅ Passed |
| `verify_integrity()` on clean log | ✅ True |
| Corrupted line → `verify_integrity()` | ✅ False (tamper detected) |
| Duplicate records (append-only) | ✅ Both present |
| Multi-user records coexist | ✅ Passed |

## Safeguard Evaluation

| Safeguard | Test | Result |
|-----------|------|--------|
| Consent gate — no consent | Access without consent | ✅ PermissionError raised |
| Consent gate — with consent | Access after granting | ✅ No exception |
| Consent gate — revoked | Access after revoking | ✅ PermissionError raised |
| AcousticModel + consent_manager, no user_id | Forward pass | ✅ PermissionError raised |
| Pipeline — too-short audio (10 samples) | process() | ✅ Returns error result |
| Pipeline — NaN audio | process() | ✅ Returns error result |
| Pipeline — empty text_input | process() | ✅ Returns empty/error gracefully |
| Speaker verifier — unregistered user | verify() | ✅ Returns (False, 0.0) |

## VAD Robustness

| Scenario | Result |
|----------|--------|
| Pure silence | ✅ 0 segments |
| Very short audio (100ms) | ✅ No crash |
| Very loud audio (clipped) | ✅ No NaN |
| Different sample rates (8k, 16k, 22050 Hz) | ✅ All work |

## Test Results

```
tests/test_robustness.py — passed
```

Sections: watermark robustness, speaker verification robustness, audit log evaluation, safeguard evaluation, VAD robustness.

## Security Findings

1. **Classical watermark is not tamper-proof**: Trimming or padding destroys the correlation-based detection. It is a provenance mark, not a cryptographic proof.
2. **Speaker verification requires trained weights for FAR/FRR measurement**: Untrained encoder cannot distinguish speakers.
3. **Audit log tamper detection is line-level**: Modifying a byte within a JSON record is detected on `verify_integrity()`. Deleting entire lines is not detected (append-only does not preclude external deletion).
4. **Consent gates enforce authorization**: All 4 tested consent scenarios passed correctly.

## Known Limitations

- Watermark breaks under temporal shifts (trim, pad, re-encode with different length)
- Speaker verification FAR/FRR not measurable without trained weights and real speaker data
- Audit log does not use cryptographic signatures (HMAC) — integrity check is structural only

## Commit

`db8fc82` — `feat(robustness): implement watermark robustness, speaker verification robustness, and safeguard evaluation tests`
