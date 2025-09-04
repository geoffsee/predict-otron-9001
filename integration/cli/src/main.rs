use std::{env, fs, io, path::PathBuf, process::Command};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

fn main() -> io::Result<()> {
    // Absolute path provided by build.rs at compile time.
    // `include_bytes!` accepts string literals; `env!` expands to a literal at compile time.
    const CLIENT_CLI: &[u8] = include_bytes!(env!("CLIENT_CLI_BIN"));

    // Write to a temp file
    let mut tmp = env::temp_dir();
    tmp.push("client-cli-embedded");

    fs::write(&tmp, CLIENT_CLI)?;

    // Ensure it's executable on Unix
    #[cfg(unix)]
    {
        let mut perms = fs::metadata(&tmp)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&tmp, perms)?;
    }

    // Run it
    let status = Command::new(&tmp).arg("--version").status()?;
    if !status.success() {
        return Err(io::Error::other("client-cli failed"));
    }

    Ok(())
}
