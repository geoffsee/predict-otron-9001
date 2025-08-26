# Root Cause Analysis: Metal error "no metal implementation for rotary-emb"

Date: 2025-08-27
Component: crates/legacy-inference-engine
Command to reproduce: crates/legacy-inference-engine/test_cli.sh

## Summary
Running the CLI with the default model (--which 3-1b-it, i.e., Gemma 3 1B Instruct) on an Apple Silicon Mac results in a runtime failure:

```
modelError: Metal error no metal implementation for rotary-emb

Caused by:
    no metal implementation for rotary-emb
```

This occurs because the project targets the Candle Metal (MPS) backend on macOS, but the Candle version in use (0.9.1) does not provide a Metal kernel implementation for the rotary embedding operation required by Gemma 3 models. The program selects the Metal device by default on macOS and hits this missing kernel during the attention computation.

## Environment and build configuration
- Machine: 2024 MacBook Pro, Apple Silicon (M4 Max)
- Crate: legacy-inference-engine
- Candle versions: pinned to =0.9.1
  - candle-core = "=0.9.1"
  - candle-transformers = "=0.9.1"
- macOS-specific dependency enabling Metal (file: crates/legacy-inference-engine/Cargo.toml):

```text
[target.'cfg(target_os = "macos")'.dependencies]
candle-core = { version = "=0.9.1", features = ["metal"] }
metal = { version = "0.32.0", features = ["mps"] }
```

- Run command (attached script): crates/legacy-inference-engine/test_cli.sh

```text
cargo run -p legacy-inference-engine --release -- --prompt 'Name the 16th President of the USA.' --which 3-1b-it
```

## What the code does at runtime
1) Device selection (defaults to Metal on macOS if available):
- File: crates/legacy-inference-engine/src/utilities_lib.rs (lines 4–12)

```text
pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        // ... falls back to CPU
        Ok(Device::Cpu)
    }
}
```

- The CLI does not pass --cpu, so on Apple Silicon with Metal available, Device::new_metal(0) is selected.

2) Default model selection is Gemma 3 1B Instruct:
- File: crates/legacy-inference-engine/src/main.rs
- Arg default (lines 705–707):

```text
/// The model to use.
#[arg(long, default_value = "3-1b-it")]
which: Which,
```

- Model id resolution (lines 758–760):

```text
Which::BaseV3_1B => "google/gemma-3-1b-pt".to_string(),
Which::InstructV3_1B => "google/gemma-3-1b-it".to_string(),
```

- Model loading uses Model3 (Gemma 3) for Which::BaseV3_1B | Which::InstructV3_1B (lines 817–821).

3) During generation, the Gemma 3 attention path requires rotary embeddings. On the Metal backend in Candle 0.9.1, the rotary embedding op is not implemented, resulting in the runtime error.

## Additional build-time signal (misleading but not causal)
- File: crates/legacy-inference-engine/src/main.rs (lines 10–11)

```text
#[cfg(feature = "metal")]
extern crate metal_src;
```

- Build warning: unexpected cfg condition value: metal
Explanation: The project does not define a Cargo feature named "metal"; instead, Metal is enabled via target-specific dependency features in Cargo.toml. This cfg gate is ineffective and triggers a warning. It does not cause the runtime failure; it just indicates confusing/obsolete gating.

## Root cause
- The program runs on the Candle Metal backend (MPS) due to device auto-selection on macOS.
- The selected model (Gemma 3 1B Instruct) requires the rotary embedding operation in its attention mechanism.
- Candle 0.9.1’s Metal backend lacks an implementation for the rotary-emb kernel. When the model executes on Metal, it attempts to invoke this operation and fails with: "no metal implementation for rotary-emb".

## Evidence
- Runtime log shows the failure immediately after model load when inference begins.
- Code paths confirm: device defaults to Metal on macOS; default model is Gemma 3; Gemma 3 uses rotary embeddings.
- Candle version pinned to 0.9.1 where rotary-emb on Metal is not available.

## Impact
- Any attempt to run Gemma 3 (and possibly other rotary-embedding reliant models) on the Metal backend with Candle 0.9.1 will fail at runtime on macOS.

## Workarounds and remediation options
1) Immediate workarounds:
- Run on CPU: add the --cpu flag to force CPU backend.
  - Example: cargo run -p legacy-inference-engine --release -- --cpu --prompt '...' --which 3-1b-it
- Use a model variant that does not hit the unimplemented kernel on Metal (e.g., older Gemma v1/v2), though many modern LLMs rely on rotary embeddings, so this may not help.

2) Recommended remediation (code/dependency changes):
- Upgrade Candle crates (candle-core, candle-transformers, etc.) to a version where the Metal backend implements rotary embeddings. Review Candle’s changelog/PRs for Metal/MPS kernel support and update to the first version that includes rotary-emb on Metal.
- Alternatively, implement a CPU fallback path for rotary-emb when running on Metal (hybrid execution). This is non-trivial and may degrade performance.
- Provide a configuration/flag to disable Metal by default on macOS for models known to require missing ops until Candle is upgraded.
- Clean up the misleading #[cfg(feature = "metal")] gate in main.rs to avoid confusion; Metal enablement is already handled in Cargo.toml via target-specific features.

## Suggested next steps
- Short term: document and expose --cpu usage in README and/or make the default model a Metal-compatible one until dependency upgrade.
- Medium term: bump Candle dependencies and test Gemma 3 on Metal; remove the obsolete cfg(feature = "metal") gate.
- Long term: integrate a device capability check and automatic fallback (informative log) when encountering unsupported kernels on the selected backend.

## References (code locations)
- crates/legacy-inference-engine/src/utilities_lib.rs lines 4–12: device selection (Metal default on macOS if available).
- crates/legacy-inference-engine/src/main.rs lines 705–707: default which = 3-1b-it.
- crates/legacy-inference-engine/src/main.rs lines 758–760 and 817–821: Gemma 3 model selection and instantiation.
- crates/legacy-inference-engine/Cargo.toml macOS target section: Candle with features = ["metal"].
- crates/legacy-inference-engine/src/main.rs lines 10–11: obsolete #[cfg(feature = "metal")] gate that triggers a warning.
