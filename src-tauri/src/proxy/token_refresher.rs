use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use crate::proxy::token_manager::{ProxyToken, TokenManager};
use crate::proxy::signature_manager::SignatureManager;

/// Token 刷新器
/// 负责定时刷新 Token，确保 API 调用的稳定性
/// 同时负责定期清理过期的签名缓存
pub struct TokenRefresher {
    /// 刷新间隔（毫秒）- 默认 5 分钟
    refresh_interval_ms: u64,
    /// 提前刷新时间（秒）- 默认 10 分钟
    refresh_ahead_secs: i64,
    /// 取消信号
    cancel_token: CancellationToken,
}

impl TokenRefresher {
    /// 创建新的 Token 刷新器
    /// 
    /// # 参数
    /// - `refresh_interval_ms`: 刷新检查间隔（毫秒），默认 300000 (5分钟)
    /// - `refresh_ahead_secs`: 提前刷新时间（秒），默认 600 (10分钟)
    pub fn new(refresh_interval_ms: u64, refresh_ahead_secs: i64) -> Self {
        Self {
            refresh_interval_ms,
            refresh_ahead_secs,
            cancel_token: CancellationToken::new(),
        }
    }

    /// 使用默认配置创建刷新器
    /// - 刷新间隔: 5 分钟
    /// - 提前刷新: 10 分钟
    pub fn with_defaults() -> Self {
        Self::new(5 * 60 * 1000, 10 * 60)
    }

    /// 启动后台刷新任务
    /// 
    /// # 参数
    /// - `token_manager`: TokenManager 的 Arc 引用
    pub fn start(&self, token_manager: Arc<TokenManager>) {
        self.start_with_signature_manager(token_manager, None);
    }

    /// 启动后台刷新任务（带签名管理器）
    /// 
    /// # 参数
    /// - `token_manager`: TokenManager 的 Arc 引用
    /// - `signature_manager`: 可选的 SignatureManager 引用，用于定期清理过期签名
    pub fn start_with_signature_manager(
        &self,
        token_manager: Arc<TokenManager>,
        signature_manager: Option<Arc<SignatureManager>>,
    ) {
        let cancel_token = self.cancel_token.clone();
        let interval_ms = self.refresh_interval_ms;
        let ahead_secs = self.refresh_ahead_secs;

        tokio::spawn(async move {
            tracing::info!(
                "Token 自动刷新任务已启动 (间隔: {}ms, 提前: {}s)",
                interval_ms,
                ahead_secs
            );

            loop {
                tokio::select! {
                    _ = cancel_token.cancelled() => {
                        tracing::info!("Token 自动刷新任务已停止");
                        break;
                    }
                    _ = tokio::time::sleep(tokio::time::Duration::from_millis(interval_ms)) => {
                        Self::refresh_all_accounts(&token_manager, ahead_secs).await;
                        
                        // 清理过期签名
                        if let Some(ref sig_manager) = signature_manager {
                            sig_manager.cleanup_expired();
                        }
                    }
                }
            }
        });
    }

    /// 停止刷新任务
    pub fn stop(&self) {
        tracing::info!("正在停止 Token 自动刷新任务...");
        self.cancel_token.cancel();
    }

    /// 检查 Token 是否需要刷新
    /// 
    /// # 参数
    /// - `token`: 要检查的 Token
    /// - `ahead_secs`: 提前刷新时间（秒）
    /// 
    /// # 返回
    /// - `true`: 需要刷新
    /// - `false`: 不需要刷新
    pub fn should_refresh(token: &ProxyToken, ahead_secs: i64) -> bool {
        let now = chrono::Utc::now().timestamp();
        // 如果当前时间 + 提前时间 >= 过期时间，则需要刷新
        now + ahead_secs >= token.timestamp
    }

    /// 刷新所有账号的 Token
    async fn refresh_all_accounts(token_manager: &TokenManager, ahead_secs: i64) {
        let tokens = token_manager.get_all_tokens();
        
        if tokens.is_empty() {
            tracing::debug!("没有需要刷新的账号");
            return;
        }

        tracing::debug!("开始检查 {} 个账号的 Token 状态", tokens.len());

        for token in tokens {
            if Self::should_refresh(&token, ahead_secs) {
                tracing::info!(
                    "账号 {} 的 Token 即将过期，正在刷新...",
                    token.email
                );

                match Self::refresh_account(&token).await {
                    Ok(new_token_response) => {
                        // 更新 TokenManager 中的 Token
                        if let Err(e) = token_manager
                            .update_token(&token.account_id, &new_token_response)
                            .await
                        {
                            tracing::error!(
                                "更新账号 {} 的 Token 失败: {}",
                                token.email,
                                e
                            );
                        } else {
                            tracing::info!(
                                "账号 {} 的 Token 刷新成功，有效期: {} 秒",
                                token.email,
                                new_token_response.expires_in
                            );
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            "刷新账号 {} 的 Token 失败: {}，将在下一周期重试",
                            token.email,
                            e
                        );
                    }
                }
            }
        }
    }

    /// 刷新单个账号的 Token
    /// 
    /// # 参数
    /// - `token`: 要刷新的 Token
    /// 
    /// # 返回
    /// - `Ok(TokenResponse)`: 刷新成功，返回新的 Token 响应
    /// - `Err(String)`: 刷新失败，返回错误信息
    async fn refresh_account(
        token: &ProxyToken,
    ) -> Result<crate::modules::oauth::TokenResponse, String> {
        crate::modules::oauth::refresh_access_token(&token.refresh_token).await
    }
}

impl Default for TokenRefresher {
    fn default() -> Self {
        Self::with_defaults()
    }
}
