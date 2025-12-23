use dashmap::DashMap;
use std::sync::Arc;

/// 签名条目
/// 存储 thoughtSignature 及其创建时间
#[derive(Clone, Debug)]
pub struct SignatureEntry {
    /// 签名内容
    pub signature: String,
    /// 创建时间戳（Unix 秒）
    pub created_at: i64,
}

impl SignatureEntry {
    /// 创建新的签名条目
    pub fn new(signature: String) -> Self {
        Self {
            signature,
            created_at: chrono::Utc::now().timestamp(),
        }
    }

    /// 检查签名是否已过期
    /// 
    /// # 参数
    /// - `expiry_secs`: 过期时间（秒）
    /// 
    /// # 返回
    /// - `true`: 已过期
    /// - `false`: 未过期
    pub fn is_expired(&self, expiry_secs: u64) -> bool {
        let now = chrono::Utc::now().timestamp();
        (now - self.created_at) as u64 > expiry_secs
    }
}

/// 签名管理器
/// 管理 Thought Signature 的缓存和恢复
/// 
/// 用于存储 Gemini API 返回的 thoughtSignature，
/// 并在后续请求中恢复对应的签名
pub struct SignatureManager {
    /// tool_use.id -> SignatureEntry 映射
    tool_signatures: Arc<DashMap<String, SignatureEntry>>,
    /// 签名过期时间（秒）- 默认 1 小时
    expiry_secs: u64,
}

impl SignatureManager {
    /// 创建新的签名管理器
    /// 
    /// # 参数
    /// - `expiry_secs`: 签名过期时间（秒）
    pub fn new(expiry_secs: u64) -> Self {
        Self {
            tool_signatures: Arc::new(DashMap::new()),
            expiry_secs,
        }
    }

    /// 使用默认配置创建签名管理器
    /// - 过期时间: 1 小时 (3600 秒)
    pub fn with_defaults() -> Self {
        Self::new(3600)
    }

    /// 存储 tool_use 签名
    /// 
    /// # 参数
    /// - `tool_use_id`: tool_use 的 ID
    /// - `signature`: thoughtSignature 内容
    pub fn store_tool_signature(&self, tool_use_id: &str, signature: &str) {
        let entry = SignatureEntry::new(signature.to_string());
        self.tool_signatures.insert(tool_use_id.to_string(), entry);
        tracing::debug!(
            "已存储 tool_use 签名: {} (当前缓存数: {})",
            tool_use_id,
            self.tool_signatures.len()
        );
    }

    /// 获取 tool_use 签名
    /// 
    /// # 参数
    /// - `tool_use_id`: tool_use 的 ID
    /// 
    /// # 返回
    /// - `Some(String)`: 找到签名
    /// - `None`: 未找到或已过期
    pub fn get_tool_signature(&self, tool_use_id: &str) -> Option<String> {
        if let Some(entry) = self.tool_signatures.get(tool_use_id) {
            if !entry.is_expired(self.expiry_secs) {
                return Some(entry.signature.clone());
            }
            // 签名已过期，移除它
            drop(entry); // 释放读锁
            self.tool_signatures.remove(tool_use_id);
            tracing::debug!("签名已过期并移除: {}", tool_use_id);
        }
        None
    }

    /// 存储通用签名（使用 responseId 或 "latest" 作为 key）
    /// 
    /// # 参数
    /// - `key`: 签名的 key（通常是 responseId 或 "latest"）
    /// - `signature`: thoughtSignature 内容
    pub fn store_signature(&self, key: &str, signature: &str) {
        let entry = SignatureEntry::new(signature.to_string());
        self.tool_signatures.insert(key.to_string(), entry);
        tracing::debug!(
            "已存储签名: {} (当前缓存数: {})",
            key,
            self.tool_signatures.len()
        );
    }

    /// 获取通用签名
    /// 
    /// # 参数
    /// - `key`: 签名的 key
    /// 
    /// # 返回
    /// - `Some(String)`: 找到签名
    /// - `None`: 未找到或已过期
    pub fn get_signature(&self, key: &str) -> Option<String> {
        self.get_tool_signature(key)
    }

    /// 获取最新的签名（key 为 "latest"）
    pub fn get_latest_signature(&self) -> Option<String> {
        self.get_signature("latest")
    }

    /// 清理过期签名
    /// 
    /// # 返回
    /// - 清理的签名数量
    pub fn cleanup_expired(&self) -> usize {
        let before_count = self.tool_signatures.len();
        
        // 收集需要删除的 key
        let expired_keys: Vec<String> = self
            .tool_signatures
            .iter()
            .filter(|entry| entry.value().is_expired(self.expiry_secs))
            .map(|entry| entry.key().clone())
            .collect();

        // 删除过期条目
        for key in &expired_keys {
            self.tool_signatures.remove(key);
        }

        let removed_count = expired_keys.len();
        if removed_count > 0 {
            tracing::info!(
                "已清理 {} 个过期签名 (剩余: {})",
                removed_count,
                self.tool_signatures.len()
            );
        } else {
            tracing::debug!(
                "无过期签名需要清理 (当前缓存数: {})",
                before_count
            );
        }

        removed_count
    }

    /// 获取当前缓存的签名数量
    pub fn len(&self) -> usize {
        self.tool_signatures.len()
    }

    /// 检查缓存是否为空
    pub fn is_empty(&self) -> bool {
        self.tool_signatures.is_empty()
    }

    /// 清空所有签名
    pub fn clear(&self) {
        self.tool_signatures.clear();
        tracing::debug!("已清空所有签名缓存");
    }
}

impl Default for SignatureManager {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl Clone for SignatureManager {
    fn clone(&self) -> Self {
        Self {
            tool_signatures: Arc::clone(&self.tool_signatures),
            expiry_secs: self.expiry_secs,
        }
    }
}
