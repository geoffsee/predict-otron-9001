# CLEANUP.md

This document tracks items requiring cleanup in the predict-otron-9000 project, identified during README updates on 2025-08-28.

## Documentation Issues

### Repository URL Inconsistencies
- **File**: `crates/inference-engine/README.md` (lines 27-28)
- **Issue**: References incorrect repository URL `https://github.com/seemueller-io/open-web-agent-rs.git`
- **Action**: Should reference the correct predict-otron-9000 repository URL
- **Priority**: High

### Model Information Discrepancies
- **File**: Main `README.md`
- **Issue**: Does not specify that inference-engine specifically uses Gemma models (1B, 2B, 7B, 9B variants)
- **Action**: Main README should clarify the specific model types supported
- **Priority**: Medium

### Build Instructions Inconsistency
- **Files**: Main `README.md` vs `crates/inference-engine/README.md`
- **Issue**: Different build commands and approaches between main and component READMEs
- **Main README**: Uses `cargo build --release` and `./run_server.sh`
- **Inference README**: Uses `cargo build -p inference-engine --release`
- **Action**: Standardize build instructions across all READMEs
- **Priority**: Medium

### Missing Component Details in Main README
- **File**: Main `README.md`
- **Issue**: Lacks specific details about:
  - Exact embedding model used (Nomic Embed Text v1.5)
  - Specific LLM models supported (Gemma variants)
  - WebAssembly nature of leptos-chat component
- **Action**: Add more specific technical details to main README
- **Priority**: Low

## Code Structure Issues

### Unified Server Reference
- **File**: Main `README.md` (line 26)
- **Issue**: Claims there's a "Main unified server that combines both engines" but unclear if this exists
- **Action**: Verify if there's actually a unified server or if this is outdated documentation
- **Priority**: Medium

### Script References
- **File**: Main `README.md`
- **Issue**: References `./run_server.sh` but needs verification that this script works as documented
- **Action**: Test and update script documentation if necessary
- **Priority**: Low

## API Documentation
- **Files**: Both READMEs
- **Issue**: API examples and endpoints should be cross-verified for accuracy
- **Action**: Ensure all API examples work with current implementation
- **Priority**: Low

## Outdated Dependencies/Versions
- **Issue**: Should verify that all mentioned Rust version requirements (1.70+) are still accurate
- **Action**: Check and update version requirements if needed
- **Priority**: Low