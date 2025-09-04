use std::env;
use std::fmt;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum BunTarget {
    LinuxX64Glibc,
    LinuxArm64Glibc,
    LinuxX64Musl,
    LinuxArm64Musl,
    WindowsX64,
    WindowsArm64,
    MacX64,
    MacArm64,
}

impl BunTarget {
    pub const fn as_bun_flag(self) -> &'static str {
        match self {
            BunTarget::LinuxX64Glibc => "bun-linux-x64",
            BunTarget::LinuxArm64Glibc => "bun-linux-arm64",
            BunTarget::LinuxX64Musl => "bun-linux-x64-musl",
            BunTarget::LinuxArm64Musl => "bun-linux-arm64-musl",
            BunTarget::WindowsX64 => "bun-windows-x64",
            BunTarget::WindowsArm64 => "bun-windows-arm64",
            BunTarget::MacX64 => "bun-darwin-x64",
            BunTarget::MacArm64 => "bun-darwin-arm64",
        }
    }

    pub const fn rust_triples(self) -> &'static [&'static str] {
        match self {
            BunTarget::LinuxX64Glibc => {
                &["x86_64-unknown-linux-gnu", "x86_64-unknown-linux-gnu.2.17"]
            }
            BunTarget::LinuxArm64Glibc => &["aarch64-unknown-linux-gnu"],
            BunTarget::LinuxX64Musl => &["x86_64-unknown-linux-musl"],
            BunTarget::LinuxArm64Musl => &["aarch64-unknown-linux-musl"],
            BunTarget::WindowsX64 => &["x86_64-pc-windows-msvc"],
            BunTarget::WindowsArm64 => &["aarch64-pc-windows-msvc"], // chart says unsupported; still map
            BunTarget::MacX64 => &["x86_64-apple-darwin"],
            BunTarget::MacArm64 => &["aarch64-apple-darwin"],
        }
    }

    pub fn from_rust_target(triple: &str) -> Option<Self> {
        let norm = triple.trim();
        if norm.starts_with("x86_64-") && norm.contains("-linux-") && norm.ends_with("gnu") {
            return Some(BunTarget::LinuxX64Glibc);
        }
        if norm.starts_with("aarch64-") && norm.contains("-linux-") && norm.ends_with("gnu") {
            return Some(BunTarget::LinuxArm64Glibc);
        }
        if norm.starts_with("x86_64-") && norm.contains("-linux-") && norm.ends_with("musl") {
            return Some(BunTarget::LinuxX64Musl);
        }
        if norm.starts_with("aarch64-") && norm.contains("-linux-") && norm.ends_with("musl") {
            return Some(BunTarget::LinuxArm64Musl);
        }
        if norm == "x86_64-pc-windows-msvc" {
            return Some(BunTarget::WindowsX64);
        }
        if norm == "aarch64-pc-windows-msvc" {
            return Some(BunTarget::WindowsArm64);
        }
        if norm == "x86_64-apple-darwin" {
            return Some(BunTarget::MacX64);
        }
        if norm == "aarch64-apple-darwin" {
            return Some(BunTarget::MacArm64);
        }
        for bt in [
            BunTarget::LinuxX64Glibc,
            BunTarget::LinuxArm64Glibc,
            BunTarget::LinuxX64Musl,
            BunTarget::LinuxArm64Musl,
            BunTarget::WindowsX64,
            BunTarget::WindowsArm64,
            BunTarget::MacX64,
            BunTarget::MacArm64,
        ] {
            for &t in bt.rust_triples() {
                if t == norm {
                    return Some(bt);
                }
            }
        }
        None
    }

    pub fn from_cargo_env() -> Result<Self, BunTargetError> {
        if let Ok(triple) = env::var("TARGET") {
            if let Some(bt) = Self::from_rust_target(&triple) {
                return Ok(bt);
            }
            return Err(BunTargetError::UnknownTriple(triple));
        }

        let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        let envv = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
        let vendor = env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_else(|_| "unknown".into());

        let triple = format!(
            "{}-{}-{}-{}",
            arch,
            vendor,
            os,
            if envv.is_empty() { "gnu" } else { &envv }
        );
        if let Some(bt) = Self::from_rust_target(&triple) {
            Ok(bt)
        } else {
            Err(BunTargetError::UnknownTriple(triple))
        }
    }
}

#[derive(Debug)]
pub enum BunTargetError {
    UnknownTriple(String),
}

impl fmt::Display for BunTargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BunTargetError::UnknownTriple(t) => write!(f, "unrecognized Rust target triple: {t}"),
        }
    }
}

impl std::error::Error for BunTargetError {}
