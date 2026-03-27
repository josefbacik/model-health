use chrono::NaiveDate;
use tracing::info;

use garmin_cli::client::api::GarminClient;
use garmin_cli::client::sso::SsoClient;
use garmin_cli::config::CredentialStore;
use garmin_cli::sync::progress::SyncMode;
use garmin_cli::{Storage, SyncEngine, SyncOptions, SyncStats};

use crate::config::Config;
use crate::error::{AppError, Result};

pub async fn authenticate() -> Result<(
    garmin_cli::client::tokens::OAuth1Token,
    garmin_cli::client::tokens::OAuth2Token,
)> {
    let creds = CredentialStore::new(None).map_err(|e| AppError::Sync(e.to_string()))?;

    if let Some((oauth1, oauth2)) = creds
        .load_tokens()
        .map_err(|e| AppError::Sync(e.to_string()))?
    {
        if !oauth2.is_expired() {
            return Ok((oauth1, oauth2));
        }
        // Try refreshing
        let sso = SsoClient::new(None).map_err(|e| AppError::Sync(e.to_string()))?;
        match sso.refresh_oauth2(&oauth1).await {
            Ok(new_token) => {
                creds
                    .save_oauth2(&new_token)
                    .map_err(|e| AppError::Sync(e.to_string()))?;
                return Ok((oauth1, new_token));
            }
            Err(_) => {
                info!("Token refresh failed, need to re-authenticate");
            }
        }
    }

    // Need fresh login - prompt for credentials
    let email = prompt("Garmin email: ")?;
    let password = prompt_password("Garmin password: ")?;

    let mut sso = SsoClient::new(None).map_err(|e| AppError::Sync(e.to_string()))?;
    let (oauth1, oauth2) = sso
        .login(
            &email,
            &password,
            Some(|| prompt("MFA code: ").unwrap_or_default()),
        )
        .await
        .map_err(|e| AppError::Sync(e.to_string()))?;

    creds
        .save_tokens(&oauth1, &oauth2)
        .map_err(|e| AppError::Sync(e.to_string()))?;
    info!("Authentication successful, tokens saved");

    Ok((oauth1, oauth2))
}

pub async fn run_sync(
    config: &Config,
    from: Option<NaiveDate>,
    to: Option<NaiveDate>,
    force: bool,
) -> Result<SyncStats> {
    let (_oauth1, oauth2) = authenticate().await?;

    let client = GarminClient::new("garmin.com");
    let storage = Storage::open(config.garmin_storage_path.clone())
        .map_err(|e| AppError::Sync(e.to_string()))?;

    let mut engine = SyncEngine::with_storage(storage, client, oauth2)
        .map_err(|e| AppError::Sync(e.to_string()))?;

    let mode = if from.is_some() {
        SyncMode::Backfill
    } else {
        SyncMode::Latest
    };

    let opts = SyncOptions {
        sync_activities: true,
        sync_health: true,
        sync_performance: true,
        from_date: from,
        to_date: to,
        force,
        mode,
        ..Default::default()
    };

    info!(?mode, ?from, ?to, force, "Starting sync");
    let stats = engine
        .run(opts)
        .await
        .map_err(|e| AppError::Sync(e.to_string()))?;
    info!(%stats, "Sync complete");

    Ok(stats)
}

fn prompt(message: &str) -> Result<String> {
    use std::io::Write;
    print!("{}", message);
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

fn prompt_password(message: &str) -> Result<String> {
    use std::io::Write;
    print!("{}", message);
    std::io::stdout().flush()?;
    // Disable echo for password input
    let password = rpassword_fallback()?;
    println!();
    Ok(password)
}

/// Simple password reading - tries to disable echo on unix
fn rpassword_fallback() -> Result<String> {
    // On unix, we can disable echo via termios
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let stdin_fd = std::io::stdin().as_raw_fd();
        let mut termios = unsafe {
            let mut t = std::mem::zeroed::<libc::termios>();
            libc::tcgetattr(stdin_fd, &mut t);
            t
        };
        let old_termios = termios;
        termios.c_lflag &= !libc::ECHO;
        unsafe { libc::tcsetattr(stdin_fd, libc::TCSANOW, &termios) };
        let mut input = String::new();
        let result = std::io::stdin().read_line(&mut input);
        unsafe { libc::tcsetattr(stdin_fd, libc::TCSANOW, &old_termios) };
        result?;
        Ok(input.trim().to_string())
    }
    #[cfg(not(unix))]
    {
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        Ok(input.trim().to_string())
    }
}
