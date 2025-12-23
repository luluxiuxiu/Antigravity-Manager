use serde_json::{json, Value};
use crate::proxy::converter::AnthropicChatRequest;

/// 默认 thinking budget (API 限制最大值 < 8192)
const DEFAULT_THINKING_BUDGET: i32 = 8191;

/// gemini-2.5-flash 的 thinking budget 限制
const FLASH_THINKING_BUDGET_LIMIT: i32 = 24576;

/// Safety Settings 的 HARM_CATEGORY 列表
const HARM_CATEGORIES: &[&str] = &[
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
];

/// 检查模型是否支持思维链功能
/// 
/// # 参数
/// - `model_name`: 模型名称
/// 
/// # 返回
/// - true: 支持思维链
/// - false: 不支持思维链
pub fn supports_thinking(model_name: &str) -> bool {
    let lower_name = model_name.to_lowercase();
    
    // 支持思维链的模型关键字
    lower_name.contains("sonnet") 
        || lower_name.contains("thinking")
        || lower_name.contains("claude-3-7")
        || lower_name.contains("opus")
        || lower_name.contains("gemini-2.5")
        || lower_name.contains("gemini-3")
}

/// 检查模型是否为 gemini-2.5-flash
/// 
/// # 参数
/// - `model_name`: 模型名称
/// 
/// # 返回
/// - true: 是 gemini-2.5-flash
/// - false: 不是
pub fn is_gemini_flash(model_name: &str) -> bool {
    let lower_name = model_name.to_lowercase();
    lower_name.contains("gemini-2.5-flash") || lower_name.contains("flash")
}

/// 构建 thinkingConfig
/// 
/// # 参数
/// - `request`: Anthropic 请求
/// - `mapped_model`: 映射后的模型名
/// 
/// # 返回
/// - Some(Value): thinkingConfig JSON 对象
/// - None: 不需要注入 thinkingConfig
pub fn build_thinking_config(request: &AnthropicChatRequest, mapped_model: &str) -> Option<Value> {
    // 检查模型是否支持思维链
    if !supports_thinking(&request.model) && !supports_thinking(mapped_model) {
        return None;
    }
    
    // 获取用户指定的 budget_tokens
    let user_budget = request.thinking
        .as_ref()
        .and_then(|t| t.budget_tokens);
    
    // 计算最终的 thinking budget
    let thinking_budget = calculate_thinking_budget(user_budget, mapped_model);
    
    Some(json!({
        "includeThoughts": true,
        "thinkingBudget": thinking_budget
    }))
}

/// 计算 thinking budget
/// 
/// # 参数
/// - `user_budget`: 用户指定的 budget (可选)
/// - `model_name`: 模型名称
/// 
/// # 返回
/// - 最终的 thinking budget 值
pub fn calculate_thinking_budget(user_budget: Option<i32>, model_name: &str) -> i32 {
    let base_budget = user_budget.unwrap_or(DEFAULT_THINKING_BUDGET);
    
    // gemini-2.5-flash 有特殊限制
    if is_gemini_flash(model_name) {
        return base_budget.min(FLASH_THINKING_BUDGET_LIMIT);
    }
    
    // 其他模型使用默认限制
    base_budget.min(DEFAULT_THINKING_BUDGET)
}

/// 构建 safetySettings 配置
/// 
/// 将所有 HARM_CATEGORY 的 threshold 设置为 OFF
/// 
/// # 返回
/// - safetySettings JSON 数组
pub fn build_safety_settings() -> Value {
    let settings: Vec<Value> = HARM_CATEGORIES
        .iter()
        .map(|category| {
            json!({
                "category": *category,
                "threshold": "OFF"
            })
        })
        .collect();
    
    Value::Array(settings)
}

/// 注入 thinkingConfig 到 generationConfig
/// 
/// # 参数
/// - `generation_config`: 现有的 generationConfig (可变引用)
/// - `request`: Anthropic 请求
/// - `mapped_model`: 映射后的模型名
pub fn inject_thinking_config(
    generation_config: &mut Value,
    request: &AnthropicChatRequest,
    mapped_model: &str,
) {
    if let Some(thinking_config) = build_thinking_config(request, mapped_model) {
        if let Some(config) = generation_config.as_object_mut() {
            config.insert("thinkingConfig".to_string(), thinking_config);
        }
    }
}

/// 构建完整的 generationConfig
/// 
/// # 参数
/// - `request`: Anthropic 请求
/// - `mapped_model`: 映射后的模型名
/// 
/// # 返回
/// - 完整的 generationConfig JSON 对象
pub fn build_generation_config(request: &AnthropicChatRequest, mapped_model: &str) -> Value {
    let mut config = json!({
        "temperature": request.temperature.unwrap_or(1.0),
        "topP": request.top_p.unwrap_or(0.95),
        "maxOutputTokens": request.max_tokens.unwrap_or(16384),
        "candidateCount": 1
    });
    
    // 注入 thinkingConfig
    inject_thinking_config(&mut config, request, mapped_model);
    
    config
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxy::converter::AnthropicThinking;

    fn create_test_request(model: &str, thinking: Option<AnthropicThinking>) -> AnthropicChatRequest {
        AnthropicChatRequest {
            model: model.to_string(),
            messages: vec![],
            system: None,
            max_tokens: Some(1000),
            metadata: None,
            stop_sequences: None,
            stream: Some(true),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: None,
            tools: None,
            thinking,
        }
    }

    #[test]
    fn test_supports_thinking() {
        // 支持思维链的模型
        assert!(supports_thinking("claude-sonnet-4-5"));
        assert!(supports_thinking("claude-sonnet-4-5-thinking"));
        assert!(supports_thinking("claude-opus-4-5-thinking"));
        assert!(supports_thinking("claude-3-7-sonnet"));
        assert!(supports_thinking("gemini-2.5-flash"));
        assert!(supports_thinking("gemini-3-pro-preview"));
        
        // 不支持思维链的模型
        assert!(!supports_thinking("gpt-4"));
        assert!(!supports_thinking("unknown-model"));
    }

    #[test]
    fn test_is_gemini_flash() {
        assert!(is_gemini_flash("gemini-2.5-flash"));
        assert!(is_gemini_flash("gemini-flash"));
        assert!(!is_gemini_flash("gemini-3-pro-preview"));
        assert!(!is_gemini_flash("claude-sonnet-4-5"));
    }

    #[test]
    fn test_calculate_thinking_budget_default() {
        // 默认值
        assert_eq!(calculate_thinking_budget(None, "claude-sonnet-4-5"), 8191);
        
        // 用户指定值 (小于限制)
        assert_eq!(calculate_thinking_budget(Some(5000), "claude-sonnet-4-5"), 5000);
        
        // 用户指定值 (大于限制)
        assert_eq!(calculate_thinking_budget(Some(10000), "claude-sonnet-4-5"), 8191);
    }

    #[test]
    fn test_calculate_thinking_budget_flash() {
        // gemini-2.5-flash 的限制
        assert_eq!(calculate_thinking_budget(None, "gemini-2.5-flash"), 8191);
        
        // 用户指定值 (小于 flash 限制)
        assert_eq!(calculate_thinking_budget(Some(20000), "gemini-2.5-flash"), 20000);
        
        // 用户指定值 (大于 flash 限制)
        assert_eq!(calculate_thinking_budget(Some(30000), "gemini-2.5-flash"), 24576);
    }

    #[test]
    fn test_build_thinking_config_with_user_budget() {
        let thinking = AnthropicThinking {
            thinking_type: "enabled".to_string(),
            budget_tokens: Some(5000),
        };
        let request = create_test_request("claude-sonnet-4-5", Some(thinking));
        
        let config = build_thinking_config(&request, "gemini-3-pro-preview");
        assert!(config.is_some());
        
        let config = config.expect("config should exist");
        assert_eq!(config["includeThoughts"], true);
        assert_eq!(config["thinkingBudget"], 5000);
    }

    #[test]
    fn test_build_thinking_config_default_budget() {
        let request = create_test_request("claude-sonnet-4-5", None);
        
        let config = build_thinking_config(&request, "gemini-3-pro-preview");
        assert!(config.is_some());
        
        let config = config.expect("config should exist");
        assert_eq!(config["includeThoughts"], true);
        assert_eq!(config["thinkingBudget"], 8191);
    }

    #[test]
    fn test_build_thinking_config_flash_limit() {
        let thinking = AnthropicThinking {
            thinking_type: "enabled".to_string(),
            budget_tokens: Some(30000),
        };
        let request = create_test_request("claude-sonnet-4-5", Some(thinking));
        
        // 使用 gemini-2.5-flash 时应该限制到 24576
        let config = build_thinking_config(&request, "gemini-2.5-flash");
        assert!(config.is_some());
        
        let config = config.expect("config should exist");
        assert_eq!(config["thinkingBudget"], 24576);
    }

    #[test]
    fn test_build_thinking_config_unsupported_model() {
        let request = create_test_request("gpt-4", None);
        
        let config = build_thinking_config(&request, "gpt-4");
        assert!(config.is_none());
    }

    #[test]
    fn test_build_safety_settings() {
        let settings = build_safety_settings();
        
        let arr = settings.as_array().expect("should be array");
        assert_eq!(arr.len(), 5);
        
        // 验证所有类别都存在且 threshold 为 OFF
        let categories: Vec<&str> = arr
            .iter()
            .filter_map(|s| s["category"].as_str())
            .collect();
        
        assert!(categories.contains(&"HARM_CATEGORY_HARASSMENT"));
        assert!(categories.contains(&"HARM_CATEGORY_HATE_SPEECH"));
        assert!(categories.contains(&"HARM_CATEGORY_SEXUALLY_EXPLICIT"));
        assert!(categories.contains(&"HARM_CATEGORY_DANGEROUS_CONTENT"));
        assert!(categories.contains(&"HARM_CATEGORY_CIVIC_INTEGRITY"));
        
        // 验证所有 threshold 都是 OFF
        for setting in arr {
            assert_eq!(setting["threshold"], "OFF");
        }
    }

    #[test]
    fn test_build_generation_config() {
        let request = create_test_request("claude-sonnet-4-5", None);
        
        let config = build_generation_config(&request, "gemini-3-pro-preview");
        
        // 使用 as_f64() 进行浮点数比较，避免精度问题
        let temp = config["temperature"].as_f64().expect("temperature should be f64");
        assert!((temp - 0.7).abs() < 0.001, "temperature should be ~0.7, got {}", temp);
        
        let top_p = config["topP"].as_f64().expect("topP should be f64");
        assert!((top_p - 0.9).abs() < 0.001, "topP should be ~0.9, got {}", top_p);
        
        assert_eq!(config["maxOutputTokens"], 1000);
        assert_eq!(config["candidateCount"], 1);
        assert!(config["thinkingConfig"].is_object());
    }
}


/// 属性测试模块
/// **Feature: anthropic-api-enhancement**
#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::proxy::converter::AnthropicThinking;
    use proptest::prelude::*;

    /// 生成随机模型名称的策略
    fn model_name_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            // 支持思维链的模型
            Just("claude-sonnet-4-5".to_string()),
            Just("claude-sonnet-4-5-thinking".to_string()),
            Just("claude-opus-4-5-thinking".to_string()),
            Just("claude-3-7-sonnet".to_string()),
            Just("gemini-2.5-flash".to_string()),
            Just("gemini-3-pro-preview".to_string()),
            // 不支持思维链的模型
            Just("gpt-4".to_string()),
            Just("unknown-model".to_string()),
            // 随机模型名
            "[a-z0-9-]{5,20}".prop_map(|s| s),
        ]
    }

    /// 生成随机 budget_tokens 的策略
    fn budget_tokens_strategy() -> impl Strategy<Value = Option<i32>> {
        prop_oneof![
            Just(None),
            (1000i32..50000i32).prop_map(Some),
        ]
    }

    fn create_test_request_with_thinking(
        model: &str,
        budget_tokens: Option<i32>,
    ) -> AnthropicChatRequest {
        let thinking = budget_tokens.map(|bt| AnthropicThinking {
            thinking_type: "enabled".to_string(),
            budget_tokens: Some(bt),
        });

        AnthropicChatRequest {
            model: model.to_string(),
            messages: vec![],
            system: None,
            max_tokens: Some(1000),
            metadata: None,
            stop_sequences: None,
            stream: Some(true),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: None,
            tools: None,
            thinking,
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Property 13: Thinking 配置注入正确性**
        /// *For any* 支持思维链的模型请求，生成的 generationConfig 应该包含 thinkingConfig，且：
        /// - includeThoughts == true
        /// - thinkingBudget == 用户指定值 或 8191（默认）
        /// - 如果模型为 gemini-2.5-flash，thinkingBudget <= 24576
        /// **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
        #[test]
        fn prop_thinking_config_injection(
            model_name in model_name_strategy(),
            budget_tokens in budget_tokens_strategy()
        ) {
            let request = create_test_request_with_thinking(&model_name, budget_tokens);
            let mapped_model = &model_name; // 简化：使用相同的模型名
            
            let thinking_config = build_thinking_config(&request, mapped_model);
            
            // 检查是否支持思维链
            let model_supports_thinking = supports_thinking(&model_name);
            
            if model_supports_thinking {
                // 支持思维链的模型应该有 thinkingConfig
                prop_assert!(thinking_config.is_some(), 
                    "Model {} should support thinking but got None", model_name);
                
                let config = thinking_config.expect("config should exist");
                
                // includeThoughts 必须为 true
                let include_thoughts = config["includeThoughts"].as_bool();
                prop_assert!(include_thoughts == Some(true),
                    "includeThoughts should be true for model {}", model_name);
                
                // 验证 thinkingBudget
                let actual_budget = config["thinkingBudget"].as_i64().expect("budget should be i64") as i32;
                
                if let Some(user_budget) = budget_tokens {
                    // 用户指定了 budget
                    if is_gemini_flash(mapped_model) {
                        // gemini-2.5-flash 限制
                        prop_assert!(actual_budget <= FLASH_THINKING_BUDGET_LIMIT,
                            "Flash model budget {} should be <= {}", actual_budget, FLASH_THINKING_BUDGET_LIMIT);
                        let expected = user_budget.min(FLASH_THINKING_BUDGET_LIMIT);
                        prop_assert!(actual_budget == expected,
                            "Flash model budget should be min(user_budget, limit), got {} expected {}", actual_budget, expected);
                    } else {
                        // 其他模型限制
                        prop_assert!(actual_budget <= DEFAULT_THINKING_BUDGET,
                            "Non-flash model budget {} should be <= {}", actual_budget, DEFAULT_THINKING_BUDGET);
                        let expected = user_budget.min(DEFAULT_THINKING_BUDGET);
                        prop_assert!(actual_budget == expected,
                            "Non-flash model budget should be min(user_budget, default_limit), got {} expected {}", actual_budget, expected);
                    }
                } else {
                    // 用户未指定 budget，使用默认值
                    prop_assert!(actual_budget == DEFAULT_THINKING_BUDGET,
                        "Default budget should be {}, got {}", DEFAULT_THINKING_BUDGET, actual_budget);
                }
            } else {
                // 不支持思维链的模型不应该有 thinkingConfig
                prop_assert!(thinking_config.is_none(),
                    "Model {} should not support thinking but got Some", model_name);
            }
        }

        /// **Property 14: Safety Settings 完整性**
        /// *For any* 生成的 Gemini 请求，safetySettings 应该包含以下五个类别，
        /// 且所有 threshold 为 "OFF"：
        /// - HARM_CATEGORY_HARASSMENT
        /// - HARM_CATEGORY_HATE_SPEECH
        /// - HARM_CATEGORY_SEXUALLY_EXPLICIT
        /// - HARM_CATEGORY_DANGEROUS_CONTENT
        /// - HARM_CATEGORY_CIVIC_INTEGRITY
        /// **Validates: Requirements 8.1, 8.2, 8.3**
        #[test]
        fn prop_safety_settings_completeness(_dummy in 0..100i32) {
            let settings = build_safety_settings();
            
            // 必须是数组
            let arr = settings.as_array();
            prop_assert!(arr.is_some(), "safetySettings should be an array");
            let arr = arr.expect("array should exist");
            
            // 必须有 5 个元素
            prop_assert!(arr.len() == 5, "safetySettings should have 5 categories, got {}", arr.len());
            
            // 收集所有类别
            let categories: Vec<&str> = arr
                .iter()
                .filter_map(|s| s["category"].as_str())
                .collect();
            
            // 验证所有必需的类别都存在
            prop_assert!(categories.contains(&"HARM_CATEGORY_HARASSMENT"),
                "Missing HARM_CATEGORY_HARASSMENT");
            prop_assert!(categories.contains(&"HARM_CATEGORY_HATE_SPEECH"),
                "Missing HARM_CATEGORY_HATE_SPEECH");
            prop_assert!(categories.contains(&"HARM_CATEGORY_SEXUALLY_EXPLICIT"),
                "Missing HARM_CATEGORY_SEXUALLY_EXPLICIT");
            prop_assert!(categories.contains(&"HARM_CATEGORY_DANGEROUS_CONTENT"),
                "Missing HARM_CATEGORY_DANGEROUS_CONTENT");
            prop_assert!(categories.contains(&"HARM_CATEGORY_CIVIC_INTEGRITY"),
                "Missing HARM_CATEGORY_CIVIC_INTEGRITY");
            
            // 验证所有 threshold 都是 "OFF"
            for setting in arr {
                let threshold = setting["threshold"].as_str();
                prop_assert!(threshold.is_some(), "threshold should be a string");
                prop_assert!(threshold == Some("OFF"),
                    "All thresholds should be OFF, got {:?}", threshold);
            }
        }
    }
}
