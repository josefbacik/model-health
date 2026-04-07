//! Authentication helpers using garmin-connect.

use tracing::info;

use garmin_connect::{CredentialStore, OAuth1Token, OAuth2Token, SsoClient};

use crate::error::{AppError, Result};

/// Authenticate with Garmin Connect, returning valid OAuth tokens.
/// Tries saved tokens first, refreshes if expired, falls back to browser login.
pub async fn authenticate() -> Result<(OAuth1Token, OAuth2Token)> {
    let creds = CredentialStore::new(None).map_err(|e| AppError::Sync(e.to_string()))?;

    // Try loading existing tokens
    if let Some((oauth1, oauth2)) = creds
        .load_tokens()
        .map_err(|e| AppError::Sync(e.to_string()))?
    {
        if !oauth2.is_expired() {
            return Ok((oauth1, oauth2));
        }
        // Try refreshing via OAuth1 (doesn't hit SSO, avoids Cloudflare)
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

    // No valid tokens — authenticate via browser
    let sso = SsoClient::new(None).map_err(|e| AppError::Sync(e.to_string()))?;
    let (oauth1, oauth2) = sso
        .login_browser()
        .await
        .map_err(|e| AppError::Sync(e.to_string()))?;

    creds
        .save_tokens(&oauth1, &oauth2)
        .map_err(|e| AppError::Sync(e.to_string()))?;
    info!("Authentication successful, tokens saved");

    Ok((oauth1, oauth2))
}
