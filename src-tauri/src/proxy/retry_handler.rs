// 重试处理模块 - 解析 429 错误响应中的 retryDelay 并决定重试策略

use regex::Regex;
use serde_json::Value;

/// 重试策略枚举
#[derive(Debug, Clone, PartialEq)]
pub enum RetryAction {
    /// 等待指定毫秒后重试同一账号
    WaitAndRetry(u64),
    /// 轮换到下一个账号
    RotateAccount,
    /// 不重试，直接返回错误
    NoRetry,
}

/// 重试延迟解析器
pub struct RetryDelayParser;

impl RetryDelayParser {
    /// 解析持续时间字符串（如 "1.203608125s", "331.167174ms", "1h16m0.667923083s"）
    /// 返回毫秒数
    pub fn parse_duration_ms(duration_str: &str) -> Option<u64> {
        if duration_str.is_empty() {
            return None;
        }
        
        let str_trimmed = duration_str.trim();
        if str_trimmed.is_empty() {
            return None;
        }
        
        let mut total_ms: f64 = 0.0;
        let mut matched = false;
        
        // 正则匹配数字+单位的组合，如 "1.5s", "200ms", "1h", "30m"
        let re = match Regex::new(r"([\d.]+)\s*(ms|s|m|h)") {
            Ok(r) => r,
            Err(_) => return None,
        };
        
        for cap in re.captures_iter(str_trimmed) {
            matched = true;
            let value: f64 = match cap.get(1).and_then(|m| m.as_str().parse().ok()) {
                Some(v) => v,
                None => continue,
            };
            
            if !value.is_finite() {
                continue;
            }
            
            let unit = cap.get(2).map(|m| m.as_str()).unwrap_or("");
            match unit {
                "ms" => total_ms += value,
                "s" => total_ms += value * 1000.0,
                "m" => total_ms += value * 60.0 * 1000.0,
                "h" => total_ms += value * 60.0 * 60.0 * 1000.0,
                _ => {}
            }
        }
        
        if !matched {
            return None;
        }
        
        Some(total_ms.round() as u64)
    }
    
    /// 解析 429 错误响应中的重试延迟
    /// 支持两种格式：
    /// 1. RetryInfo.retryDelay (如 "1.203608125s")
    /// 2. metadata.quotaResetDelay (如 "331.167174ms" 或 "1h16m0.667923083s")
    pub fn parse_retry_delay_ms(error_text: &str) -> Option<u64> {
        // 尝试解析 JSON
        let err_obj: Value = match serde_json::from_str(error_text) {
            Ok(v) => v,
            Err(_) => return None,
        };
        
        // 获取 error.details 数组
        let details = err_obj
            .get("error")
            .and_then(|e| e.get("details"))
            .and_then(|d| d.as_array());
        
        let details = match details {
            Some(d) => d,
            None => return None,
        };
        
        // 1. 查找 RetryInfo.retryDelay
        for detail in details {
            let type_field = detail.get("@type").and_then(|t| t.as_str()).unwrap_or("");
            if type_field.contains("RetryInfo") {
                if let Some(retry_delay) = detail.get("retryDelay").and_then(|d| d.as_str()) {
                    if let Some(ms) = Self::parse_duration_ms(retry_delay) {
                        return Some(ms);
                    }
                }
            }
        }
        
        // 2. 查找 metadata.quotaResetDelay
        for detail in details {
            if let Some(metadata) = detail.get("metadata") {
                if let Some(quota_reset_delay) = metadata.get("quotaResetDelay").and_then(|d| d.as_str()) {
                    if let Some(ms) = Self::parse_duration_ms(quota_reset_delay) {
                        return Some(ms);
                    }
                }
            }
        }
        
        None
    }
    
    /// 决定重试策略
    /// 
    /// 规则：
    /// - 429 且 retryDelay <= 5000ms：等待后重试同一账号
    /// - 429 且 retryDelay > 5000ms 或无法解析：轮换账号
    /// - 404 或 403：轮换账号
    /// - 其他：不重试
    pub fn decide_retry_action(status: u16, error_text: &str) -> RetryAction {
        if status == 429 {
            if let Some(delay_ms) = Self::parse_retry_delay_ms(error_text) {
                if delay_ms <= 5000 {
                    // 加 200ms 缓冲
                    return RetryAction::WaitAndRetry(delay_ms + 200);
                }
            }
            return RetryAction::RotateAccount;
        }
        
        if status == 404 || status == 403 {
            return RetryAction::RotateAccount;
        }
        
        RetryAction::NoRetry
    }
    
    /// 检查是否应该因为空响应而重试
    /// 
    /// 规则：如果响应内容为空且 finish_reason 为 MAX_TOKENS 或 STOP，应该触发重试
    pub fn should_retry_empty_response(content: &str, finish_reason: Option<&str>) -> bool {
        if !content.is_empty() {
            return false;
        }
        
        match finish_reason {
            Some("MAX_TOKENS") | Some("STOP") | Some("max_tokens") | Some("stop") | Some("length") => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_duration_ms_seconds() {
        assert_eq!(RetryDelayParser::parse_duration_ms("1.203608125s"), Some(1204));
        assert_eq!(RetryDelayParser::parse_duration_ms("0.5s"), Some(500));
        assert_eq!(RetryDelayParser::parse_duration_ms("2s"), Some(2000));
    }
    
    #[test]
    fn test_parse_duration_ms_milliseconds() {
        assert_eq!(RetryDelayParser::parse_duration_ms("331.167174ms"), Some(331));
        assert_eq!(RetryDelayParser::parse_duration_ms("500ms"), Some(500));
        assert_eq!(RetryDelayParser::parse_duration_ms("1000ms"), Some(1000));
    }
    
    #[test]
    fn test_parse_duration_ms_complex() {
        // 1h16m0.667923083s = 3600000 + 960000 + 668 = 4560668ms
        assert_eq!(RetryDelayParser::parse_duration_ms("1h16m0.667923083s"), Some(4560668));
        assert_eq!(RetryDelayParser::parse_duration_ms("1h"), Some(3600000));
        assert_eq!(RetryDelayParser::parse_duration_ms("30m"), Some(1800000));
    }
    
    #[test]
    fn test_parse_duration_ms_invalid() {
        assert_eq!(RetryDelayParser::parse_duration_ms(""), None);
        assert_eq!(RetryDelayParser::parse_duration_ms("   "), None);
        assert_eq!(RetryDelayParser::parse_duration_ms("invalid"), None);
        assert_eq!(RetryDelayParser::parse_duration_ms("123"), None);
    }
    
    #[test]
    fn test_parse_retry_delay_ms_retry_info() {
        let error_json = r#"{
            "error": {
                "code": 429,
                "message": "Resource exhausted",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "1.5s"
                    }
                ]
            }
        }"#;
        assert_eq!(RetryDelayParser::parse_retry_delay_ms(error_json), Some(1500));
    }
    
    #[test]
    fn test_parse_retry_delay_ms_quota_reset() {
        let error_json = r#"{
            "error": {
                "code": 429,
                "message": "Quota exceeded",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "metadata": {
                            "quotaResetDelay": "331.167174ms"
                        }
                    }
                ]
            }
        }"#;
        assert_eq!(RetryDelayParser::parse_retry_delay_ms(error_json), Some(331));
    }
    
    #[test]
    fn test_parse_retry_delay_ms_no_details() {
        let error_json = r#"{"error": {"code": 429, "message": "Rate limited"}}"#;
        assert_eq!(RetryDelayParser::parse_retry_delay_ms(error_json), None);
    }
    
    #[test]
    fn test_decide_retry_action_429_short_delay() {
        let error_json = r#"{
            "error": {
                "code": 429,
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "1.5s"
                    }
                ]
            }
        }"#;
        assert_eq!(
            RetryDelayParser::decide_retry_action(429, error_json),
            RetryAction::WaitAndRetry(1700) // 1500 + 200
        );
    }
    
    #[test]
    fn test_decide_retry_action_429_long_delay() {
        let error_json = r#"{
            "error": {
                "code": 429,
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "10s"
                    }
                ]
            }
        }"#;
        assert_eq!(
            RetryDelayParser::decide_retry_action(429, error_json),
            RetryAction::RotateAccount
        );
    }
    
    #[test]
    fn test_decide_retry_action_429_no_delay() {
        let error_json = r#"{"error": {"code": 429, "message": "Rate limited"}}"#;
        assert_eq!(
            RetryDelayParser::decide_retry_action(429, error_json),
            RetryAction::RotateAccount
        );
    }
    
    #[test]
    fn test_decide_retry_action_404() {
        assert_eq!(
            RetryDelayParser::decide_retry_action(404, "Not found"),
            RetryAction::RotateAccount
        );
    }
    
    #[test]
    fn test_decide_retry_action_403() {
        assert_eq!(
            RetryDelayParser::decide_retry_action(403, "Permission denied"),
            RetryAction::RotateAccount
        );
    }
    
    #[test]
    fn test_decide_retry_action_500() {
        assert_eq!(
            RetryDelayParser::decide_retry_action(500, "Internal error"),
            RetryAction::NoRetry
        );
    }
    
    #[test]
    fn test_should_retry_empty_response() {
        // 空内容 + MAX_TOKENS -> 重试
        assert!(RetryDelayParser::should_retry_empty_response("", Some("MAX_TOKENS")));
        assert!(RetryDelayParser::should_retry_empty_response("", Some("max_tokens")));
        
        // 空内容 + STOP -> 重试
        assert!(RetryDelayParser::should_retry_empty_response("", Some("STOP")));
        assert!(RetryDelayParser::should_retry_empty_response("", Some("stop")));
        assert!(RetryDelayParser::should_retry_empty_response("", Some("length")));
        
        // 有内容 -> 不重试
        assert!(!RetryDelayParser::should_retry_empty_response("Hello", Some("STOP")));
        
        // 空内容但无 finish_reason -> 不重试
        assert!(!RetryDelayParser::should_retry_empty_response("", None));
        
        // 空内容但其他 finish_reason -> 不重试
        assert!(!RetryDelayParser::should_retry_empty_response("", Some("SAFETY")));
    }
}

// ===== 属性测试 =====
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    
    /// 生成有效的持续时间字符串
    fn duration_string_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            // 毫秒格式
            (1u64..10000).prop_map(|ms| format!("{}ms", ms)),
            // 秒格式
            (1u64..100).prop_map(|s| format!("{}s", s)),
            // 带小数的秒格式
            (1u64..100, 0u64..999999).prop_map(|(s, frac)| format!("{}.{}s", s, frac)),
            // 分钟格式
            (1u64..60).prop_map(|m| format!("{}m", m)),
            // 小时格式
            (1u64..24).prop_map(|h| format!("{}h", h)),
            // 复合格式
            (0u64..24, 0u64..60, 0u64..60).prop_map(|(h, m, s)| format!("{}h{}m{}s", h, m, s)),
        ]
    }
    
    /// 生成 429 错误响应 JSON
    fn error_json_with_delay_strategy(delay_ms: u64) -> String {
        // 使用毫秒格式以确保精确性
        let delay_str = format!("{}ms", delay_ms);
        
        format!(r#"{{
            "error": {{
                "code": 429,
                "message": "Resource exhausted",
                "details": [
                    {{
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "{}"
                    }}
                ]
            }}
        }}"#, delay_str)
    }
    
    proptest! {
        /// Property 9: 重试策略决策正确性
        /// 
        /// *For any* HTTP 状态码和错误响应体：
        /// - 如果状态码为 429 且 retryDelay <= 5000ms，应该返回 WaitAndRetry 策略
        /// - 如果状态码为 429 且 retryDelay > 5000ms 或无法解析，应该返回 RotateAccount 策略
        /// - 如果状态码为 404 或 403，应该返回 RotateAccount 策略
        /// - 其他情况应该返回 NoRetry 策略
        /// 
        /// **Validates: Requirements 5.1, 5.2, 5.3**
        #[test]
        fn prop_retry_action_429_short_delay(delay_ms in 1u64..5000) {
            // Feature: anthropic-api-enhancement, Property 9: 重试策略决策正确性
            let error_json = error_json_with_delay_strategy(delay_ms);
            let action = RetryDelayParser::decide_retry_action(429, &error_json);
            
            // 429 + 短延迟 (<=5000ms) 应该返回 WaitAndRetry
            match action {
                RetryAction::WaitAndRetry(wait_ms) => {
                    // 等待时间应该是 delay_ms + 200 (缓冲)
                    // 由于解析可能有精度损失，允许一定误差
                    let expected_min = delay_ms;
                    let expected_max = delay_ms + 300; // 200 缓冲 + 100 误差
                    prop_assert!(
                        wait_ms >= expected_min && wait_ms <= expected_max,
                        "等待时间 {} 不在预期范围 [{}, {}]", wait_ms, expected_min, expected_max
                    );
                },
                other => prop_assert!(false, "429 短延迟应返回 WaitAndRetry，实际返回 {:?}", other),
            }
        }
        
        #[test]
        fn prop_retry_action_429_long_delay(delay_ms in 5001u64..100000) {
            // Feature: anthropic-api-enhancement, Property 9: 重试策略决策正确性
            let error_json = error_json_with_delay_strategy(delay_ms);
            let action = RetryDelayParser::decide_retry_action(429, &error_json);
            
            // 429 + 长延迟 (>5000ms) 应该返回 RotateAccount
            prop_assert_eq!(
                action, 
                RetryAction::RotateAccount,
                "429 长延迟应返回 RotateAccount"
            );
        }
        
        #[test]
        fn prop_retry_action_404_403(status in prop_oneof![Just(404u16), Just(403u16)]) {
            // Feature: anthropic-api-enhancement, Property 9: 重试策略决策正确性
            let action = RetryDelayParser::decide_retry_action(status, "any error text");
            
            // 404/403 应该返回 RotateAccount
            prop_assert_eq!(
                action,
                RetryAction::RotateAccount,
                "{} 应返回 RotateAccount", status
            );
        }
        
        #[test]
        fn prop_retry_action_other_status(status in 200u16..600) {
            // Feature: anthropic-api-enhancement, Property 9: 重试策略决策正确性
            // 排除 429, 404, 403
            prop_assume!(status != 429 && status != 404 && status != 403);
            
            let action = RetryDelayParser::decide_retry_action(status, "any error text");
            
            // 其他状态码应该返回 NoRetry
            prop_assert_eq!(
                action,
                RetryAction::NoRetry,
                "状态码 {} 应返回 NoRetry", status
            );
        }
        
        /// Property 10: 空响应重试触发
        /// 
        /// *For any* 响应，如果内容为空且 finish_reason 为 "MAX_TOKENS" 或 "STOP"，应该触发重试。
        /// 
        /// **Validates: Requirements 5.6**
        #[test]
        fn prop_empty_response_retry_trigger(
            finish_reason in prop_oneof![
                Just("MAX_TOKENS"),
                Just("STOP"),
                Just("max_tokens"),
                Just("stop"),
                Just("length")
            ]
        ) {
            // Feature: anthropic-api-enhancement, Property 10: 空响应重试触发
            // 空内容 + 特定 finish_reason 应该触发重试
            prop_assert!(
                RetryDelayParser::should_retry_empty_response("", Some(finish_reason)),
                "空内容 + {} 应触发重试", finish_reason
            );
        }
        
        #[test]
        fn prop_non_empty_response_no_retry(content in "[a-zA-Z0-9]{1,100}") {
            // Feature: anthropic-api-enhancement, Property 10: 空响应重试触发
            // 有内容时不应该触发重试，无论 finish_reason 是什么
            for reason in &["MAX_TOKENS", "STOP", "SAFETY", "OTHER"] {
                prop_assert!(
                    !RetryDelayParser::should_retry_empty_response(&content, Some(reason)),
                    "有内容时不应触发重试 (content={}, reason={})", content, reason
                );
            }
        }
        
        #[test]
        fn prop_empty_response_other_reason_no_retry(
            finish_reason in prop_oneof![
                Just("SAFETY"),
                Just("RECITATION"),
                Just("OTHER"),
                Just("BLOCKED")
            ]
        ) {
            // Feature: anthropic-api-enhancement, Property 10: 空响应重试触发
            // 空内容 + 其他 finish_reason 不应该触发重试
            prop_assert!(
                !RetryDelayParser::should_retry_empty_response("", Some(finish_reason)),
                "空内容 + {} 不应触发重试", finish_reason
            );
        }
        
        /// 持续时间解析往返测试
        #[test]
        fn prop_duration_parse_positive(duration_str in duration_string_strategy()) {
            // 有效的持续时间字符串应该能被解析
            let result = RetryDelayParser::parse_duration_ms(&duration_str);
            prop_assert!(
                result.is_some(),
                "有效持续时间字符串 '{}' 应能被解析", duration_str
            );
            
            // 解析结果应该是正数
            if let Some(ms) = result {
                prop_assert!(ms > 0, "解析结果应为正数，实际为 {}", ms);
            }
        }
    }
}
