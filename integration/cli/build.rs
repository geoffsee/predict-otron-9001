use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStderr, ChildStdout, Command, Stdio};
use std::thread;
use std::time::{Duration, SystemTime};
mod bun_target;
use bun_target::BunTarget;

fn main() {
    println!("cargo:rerun-if-changed=");

    if let Err(e) = run_build() {
        println!("cargo:warning=build.rs failed: {e}");
        std::process::exit(1);
    }
}

fn run_build() -> io::Result<()> {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let package_dir = manifest_dir.join("package");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set by Cargo"));
    let output_path = out_dir.join("client-cli");

    let bun_tgt = BunTarget::from_cargo_env().map_err(|e| io::Error::other(e.to_string()))?;

    // Optional: warn if using a Bun target that’s marked unsupported in your chart
    if matches!(bun_tgt, BunTarget::WindowsArm64) {
        println!(
            "cargo:warning=bun-windows-arm64 is marked unsupported in the compatibility chart"
        );
    }

    warn(&format!("Building CLI into: {}", output_path.display()));

    // --- bun install (in ./package), keep temps inside OUT_DIR ---
    let mut install = Command::new("bun")
        .current_dir(&package_dir)
        .env("TMPDIR", &out_dir)
        .arg("install")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| io::Error::new(e.kind(), format!("Failed to spawn `bun install`: {e}")))?;

    let install_join = stream_child("bun install", install.stdout.take(), install.stderr.take());
    let install_status = install.wait()?;
    // ensure streams finish
    join_streams(install_join);

    if !install_status.success() {
        let code = install_status.code().unwrap_or(1);
        return Err(io::Error::other(format!(
            "bun install failed with status {code}"
        )));
    }

    let _target = env::var("TARGET").unwrap();

    // --- bun build (in ./package), emit to OUT_DIR, keep temps inside OUT_DIR ---
    let mut build = Command::new("bun")
        .current_dir(&package_dir)
        .env("TMPDIR", &out_dir)
        .arg("build")
        .arg("./cli.ts")
        .arg(format!("--target={}", bun_tgt.as_bun_flag()))
        .arg("--compile")
        .arg("--outfile")
        .arg(&output_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| io::Error::new(e.kind(), format!("Failed to spawn `bun build`: {e}")))?;

    let build_join = stream_child("bun build", build.stdout.take(), build.stderr.take());
    let status = build.wait()?;
    // ensure streams finish
    join_streams(build_join);

    if status.success() {
        info("bun build succeeded");
    } else {
        let code = status.code().unwrap_or(1);
        warn(&format!("bun build failed with status: {code}"));
        return Err(io::Error::other("bun build failed"));
    }

    // Ensure the output is executable (after it exists)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&output_path)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&output_path, perms)?;
    }

    println!("cargo:warning=Built CLI at {}", output_path.display());
    println!("cargo:rustc-env=CLIENT_CLI_BIN={}", output_path.display());

    // --- Cleanup stray .bun-build temp files (conservative: older than 5 minutes) ---
    for dir in [&manifest_dir, &package_dir, &out_dir] {
        if let Err(e) = remove_bun_temp_files(dir, Some(Duration::from_secs(5 * 60))) {
            println!("cargo:warning=cleanup in {} failed: {e}", dir.display());
        }
    }

    Ok(())
}

// Spawn readers for child's stdout/stderr so we don't deadlock on pipe buffers
fn stream_child(
    tag: &str,
    stdout: Option<ChildStdout>,
    stderr: Option<ChildStderr>,
) -> (
    Option<thread::JoinHandle<()>>,
    Option<thread::JoinHandle<()>>,
) {
    let t1 = stdout.map(|out| {
        let tag = tag.to_string();
        thread::spawn(move || {
            let reader = io::BufReader::new(out);
            for line in reader.lines() {
                info(&format!("[{tag} stdout] {}", line.unwrap_or_default()));
            }
        })
    });
    let t2 = stderr.map(|err| {
        let tag = tag.to_string();
        thread::spawn(move || {
            let reader = io::BufReader::new(err);
            for line in reader.lines() {
                warn(&format!("[{tag} stderr] {}", line.unwrap_or_default()));
            }
        })
    });
    (t1, t2)
}

fn join_streams(
    joins: (
        Option<thread::JoinHandle<()>>,
        Option<thread::JoinHandle<()>>,
    ),
) {
    if let Some(j) = joins.0 {
        let _ = j.join();
    }
    if let Some(j) = joins.1 {
        let _ = j.join();
    }
}

fn remove_bun_temp_files(dir: &Path, older_than: Option<Duration>) -> io::Result<()> {
    let now = SystemTime::now();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        // Files like ".1860e7df40ff1bef-00000000.bun-build"
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let looks_like_bun_temp = name.starts_with('.') && name.ends_with(".bun-build");

        if !looks_like_bun_temp {
            continue;
        }

        if let Some(age) = older_than {
            if let Ok(meta) = entry.metadata() {
                if let Ok(modified) = meta.modified() {
                    if now.duration_since(modified).unwrap_or_default() < age {
                        // too new; skip to avoid racing an in-flight builder
                        continue;
                    }
                }
            }
        }

        match fs::remove_file(&path) {
            Ok(_) => println!("cargo:warning=removed stray bun temp {}", path.display()),
            Err(e) => println!("cargo:warning=failed to remove {}: {e}", path.display()),
        }
    }
    Ok(())
}

fn warn(msg: &str) {
    let _ = writeln!(io::stderr(), "[build.rs] {msg}");
    println!("cargo:warning={msg}");
}

fn info(msg: &str) {
    let _ = writeln!(io::stderr(), "[build.rs] {msg}");
    println!("cargo:warning=INFO|{msg}");
}
