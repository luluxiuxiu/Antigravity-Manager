use std::collections::HashMap;

/// 模型映射器
/// 负责将 Claude/Anthropic 模型名映射到 Gemini 模型名
pub struct ModelMapper {
    /// 用户自定义映射 (优先级最高)
    custom_mapping: HashMap<String, String>,
}

/// 默认支持的模型列表 (直接透传)
const SUPPORTED_MODELS: &[&str] = &[
    "claude-opus-4-5-thinking",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-thinking",
    "gemini-2.5-flash",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.0-flash-exp",
];

impl ModelMapper {
    /// 创建新的模型映射器
    pub fn new(custom_mapping: HashMap<String, String>) -> Self {
        Self { custom_mapping }
    }

    /// 创建空的模型映射器 (无自定义映射)
    pub fn empty() -> Self {
        Self {
            custom_mapping: HashMap::new(),
        }
    }

    /// 映射模型名称
    /// 优先级: 用户自定义映射 > 默认映射规则
    pub fn map_model_name(&self, model_name: &str) -> String {
        // 1. 用户自定义映射优先
        if let Some(mapped) = self.custom_mapping.get(model_name) {
            return mapped.clone();
        }

        // 2. 如果是已支持的模型，直接返回
        if SUPPORTED_MODELS.contains(&model_name) {
            return model_name.to_string();
        }

        // 3. 默认映射规则
        Self::default_mapping(model_name)
    }

    /// 检查 tools 中是否包含 web_search 工具
    pub fn has_web_search_tool(tools: Option<&Vec<serde_json::Value>>) -> bool {
        if let Some(tools) = tools {
            return tools.iter().any(|tool| {
                tool.get("name")
                    .and_then(|n| n.as_str())
                    .map(|n| n == "web_search")
                    .unwrap_or(false)
            });
        }
        false
    }

    /// 映射模型名称，考虑 web_search 工具强制
    /// 如果包含 web_search 工具，强制使用 gemini-2.5-flash
    pub fn map_model_with_tools(
        &self,
        model_name: &str,
        tools: Option<&Vec<serde_json::Value>>,
    ) -> String {
        // web_search 工具强制使用 gemini-2.5-flash
        if Self::has_web_search_tool(tools) {
            return "gemini-2.5-flash".to_string();
        }

        self.map_model_name(model_name)
    }

    /// 默认映射规则
    fn default_mapping(model_name: &str) -> String {
        // 精确匹配映射表
        let exact_mapping: HashMap<&str, &str> = [
            // Claude Sonnet 系列
            ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5-thinking"),
            ("claude-3-5-sonnet-20241022", "claude-sonnet-4-5"),
            ("claude-3-5-sonnet-20240620", "claude-sonnet-4-5"),
            // Claude Opus 系列
            ("claude-opus-4", "claude-opus-4-5-thinking"),
            ("claude-opus-4-5-20251101", "claude-opus-4-5-thinking"),
            ("claude-opus-4-5", "claude-opus-4-5-thinking"),
            // Claude Haiku 系列
            ("claude-haiku-4", "claude-sonnet-4-5"),
            ("claude-3-haiku-20240307", "claude-sonnet-4-5"),
            ("claude-haiku-4-5-20251001", "claude-sonnet-4-5"),
            // Gemini 内部映射
            ("gemini-3-pro-high", "gemini-3-pro-preview"),
            ("gemini-3-pro-low", "gemini-3-pro-preview"),
            ("gemini-3-flash", "gemini-3-flash-preview"),
        ]
        .into_iter()
        .collect();

        // 1. 精确匹配
        if let Some(mapped) = exact_mapping.get(model_name) {
            return mapped.to_string();
        }

        let lower_name = model_name.to_lowercase();

        // 2. 模糊匹配规则
        // sonnet 或 thinking -> gemini-3-pro-preview
        if lower_name.contains("sonnet") || lower_name.contains("thinking") {
            return "gemini-3-pro-preview".to_string();
        }

        // haiku -> gemini-2.0-flash-exp
        if lower_name.contains("haiku") {
            return "gemini-2.0-flash-exp".to_string();
        }

        // opus -> gemini-3-pro-preview
        if lower_name.contains("opus") {
            return "gemini-3-pro-preview".to_string();
        }

        // 3. 如果已经是 gemini 模型，直接返回
        if lower_name.starts_with("gemini-") {
            return model_name.to_string();
        }

        // 4. 默认回退到 claude-sonnet-4-5
        "claude-sonnet-4-5".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_mapping() {
        let mapper = ModelMapper::empty();

        // Claude Sonnet 系列
        assert_eq!(
            mapper.map_model_name("claude-sonnet-4-5-20250929"),
            "claude-sonnet-4-5-thinking"
        );
        assert_eq!(
            mapper.map_model_name("claude-3-5-sonnet-20241022"),
            "claude-sonnet-4-5"
        );
        assert_eq!(
            mapper.map_model_name("claude-3-5-sonnet-20240620"),
            "claude-sonnet-4-5"
        );

        // Claude Opus 系列
        assert_eq!(
            mapper.map_model_name("claude-opus-4"),
            "claude-opus-4-5-thinking"
        );
        assert_eq!(
            mapper.map_model_name("claude-opus-4-5"),
            "claude-opus-4-5-thinking"
        );

        // Claude Haiku 系列
        assert_eq!(mapper.map_model_name("claude-haiku-4"), "claude-sonnet-4-5");
        assert_eq!(
            mapper.map_model_name("claude-3-haiku-20240307"),
            "claude-sonnet-4-5"
        );

        // Gemini 内部映射
        assert_eq!(
            mapper.map_model_name("gemini-3-pro-high"),
            "gemini-3-pro-preview"
        );
        assert_eq!(
            mapper.map_model_name("gemini-3-pro-low"),
            "gemini-3-pro-preview"
        );
    }

    #[test]
    fn test_fuzzy_mapping() {
        let mapper = ModelMapper::empty();

        // sonnet 关键字
        assert_eq!(
            mapper.map_model_name("some-sonnet-model"),
            "gemini-3-pro-preview"
        );

        // thinking 关键字
        assert_eq!(
            mapper.map_model_name("claude-thinking-v2"),
            "gemini-3-pro-preview"
        );

        // haiku 关键字
        assert_eq!(
            mapper.map_model_name("claude-haiku-new"),
            "gemini-2.0-flash-exp"
        );

        // opus 关键字
        assert_eq!(
            mapper.map_model_name("claude-opus-new"),
            "gemini-3-pro-preview"
        );
    }

    #[test]
    fn test_supported_models_passthrough() {
        let mapper = ModelMapper::empty();

        // 已支持的模型直接透传
        assert_eq!(
            mapper.map_model_name("claude-opus-4-5-thinking"),
            "claude-opus-4-5-thinking"
        );
        assert_eq!(
            mapper.map_model_name("claude-sonnet-4-5"),
            "claude-sonnet-4-5"
        );
        assert_eq!(
            mapper.map_model_name("gemini-2.5-flash"),
            "gemini-2.5-flash"
        );
    }

    #[test]
    fn test_custom_mapping_priority() {
        let mut custom = HashMap::new();
        custom.insert(
            "my-custom-model".to_string(),
            "gemini-custom".to_string(),
        );
        custom.insert(
            "claude-sonnet-4-5".to_string(),
            "my-override".to_string(),
        );

        let mapper = ModelMapper::new(custom);

        // 自定义映射优先
        assert_eq!(mapper.map_model_name("my-custom-model"), "gemini-custom");
        // 覆盖默认支持的模型
        assert_eq!(mapper.map_model_name("claude-sonnet-4-5"), "my-override");
    }

    #[test]
    fn test_gemini_passthrough() {
        let mapper = ModelMapper::empty();

        // gemini 模型直接透传
        assert_eq!(
            mapper.map_model_name("gemini-3-pro-preview"),
            "gemini-3-pro-preview"
        );
        assert_eq!(
            mapper.map_model_name("gemini-unknown-model"),
            "gemini-unknown-model"
        );
    }

    #[test]
    fn test_default_fallback() {
        let mapper = ModelMapper::empty();

        // 未知模型回退到 claude-sonnet-4-5
        assert_eq!(
            mapper.map_model_name("unknown-model"),
            "claude-sonnet-4-5"
        );
    }

    #[test]
    fn test_web_search_detection() {
        // 无 tools
        assert!(!ModelMapper::has_web_search_tool(None));

        // 空 tools
        let empty_tools: Vec<serde_json::Value> = vec![];
        assert!(!ModelMapper::has_web_search_tool(Some(&empty_tools)));

        // 有 web_search
        let tools_with_web_search: Vec<serde_json::Value> = vec![
            serde_json::json!({"name": "calculator"}),
            serde_json::json!({"name": "web_search"}),
        ];
        assert!(ModelMapper::has_web_search_tool(Some(&tools_with_web_search)));

        // 无 web_search
        let tools_without_web_search: Vec<serde_json::Value> = vec![
            serde_json::json!({"name": "calculator"}),
            serde_json::json!({"name": "file_reader"}),
        ];
        assert!(!ModelMapper::has_web_search_tool(Some(
            &tools_without_web_search
        )));
    }

    #[test]
    fn test_web_search_forces_flash() {
        let mapper = ModelMapper::empty();

        let tools_with_web_search: Vec<serde_json::Value> =
            vec![serde_json::json!({"name": "web_search"})];

        // 任何模型 + web_search 都应该返回 gemini-2.5-flash
        assert_eq!(
            mapper.map_model_with_tools("claude-opus-4-5-thinking", Some(&tools_with_web_search)),
            "gemini-2.5-flash"
        );
        assert_eq!(
            mapper.map_model_with_tools("claude-sonnet-4-5", Some(&tools_with_web_search)),
            "gemini-2.5-flash"
        );
        assert_eq!(
            mapper.map_model_with_tools("gemini-3-pro-preview", Some(&tools_with_web_search)),
            "gemini-2.5-flash"
        );

        // 无 web_search 时正常映射
        let tools_without_web_search: Vec<serde_json::Value> =
            vec![serde_json::json!({"name": "calculator"})];
        assert_eq!(
            mapper.map_model_with_tools("claude-opus-4-5-thinking", Some(&tools_without_web_search)),
            "claude-opus-4-5-thinking"
        );
    }
}

/// 属性测试模块
/// **Feature: anthropic-api-enhancement**
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    /// 生成随机模型名称的策略
    fn model_name_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            // 已知的 Claude 模型名
            Just("claude-sonnet-4-5".to_string()),
            Just("claude-sonnet-4-5-thinking".to_string()),
            Just("claude-opus-4-5-thinking".to_string()),
            Just("claude-sonnet-4-5-20250929".to_string()),
            Just("claude-3-5-sonnet-20241022".to_string()),
            Just("claude-3-5-sonnet-20240620".to_string()),
            Just("claude-opus-4".to_string()),
            Just("claude-opus-4-5".to_string()),
            Just("claude-haiku-4".to_string()),
            Just("claude-3-haiku-20240307".to_string()),
            // Gemini 模型名
            Just("gemini-3-pro-preview".to_string()),
            Just("gemini-3-pro-high".to_string()),
            Just("gemini-3-pro-low".to_string()),
            Just("gemini-2.5-flash".to_string()),
            Just("gemini-2.0-flash-exp".to_string()),
            // 包含关键字的随机模型名
            "[a-z]{3,8}-sonnet-[a-z0-9]{3,6}".prop_map(|s| s),
            "[a-z]{3,8}-haiku-[a-z0-9]{3,6}".prop_map(|s| s),
            "[a-z]{3,8}-opus-[a-z0-9]{3,6}".prop_map(|s| s),
            "[a-z]{3,8}-thinking-[a-z0-9]{3,6}".prop_map(|s| s),
            // 完全随机的模型名
            "[a-z0-9-]{5,20}".prop_map(|s| s),
        ]
    }

    /// 生成随机工具列表的策略
    fn tools_strategy() -> impl Strategy<Value = Option<Vec<serde_json::Value>>> {
        prop_oneof![
            // 无工具
            Just(None),
            // 空工具列表
            Just(Some(vec![])),
            // 包含 web_search 的工具列表
            Just(Some(vec![
                serde_json::json!({"name": "web_search"}),
            ])),
            Just(Some(vec![
                serde_json::json!({"name": "calculator"}),
                serde_json::json!({"name": "web_search"}),
            ])),
            // 不包含 web_search 的工具列表
            Just(Some(vec![
                serde_json::json!({"name": "calculator"}),
            ])),
            Just(Some(vec![
                serde_json::json!({"name": "file_reader"}),
                serde_json::json!({"name": "code_executor"}),
            ])),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Property 11: 模型映射正确性**
        /// *For any* 模型名称：
        /// - 精确匹配映射表优先
        /// - 包含 "sonnet" 或 "thinking" -> "gemini-3-pro-preview"
        /// - 包含 "haiku" -> "gemini-2.0-flash-exp"
        /// - 包含 "opus" -> "gemini-3-pro-preview"
        /// - 等于 "gemini-3-pro-high" 或 "gemini-3-pro-low" -> "gemini-3-pro-preview"
        /// - 用户自定义映射优先于默认映射
        /// **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
        #[test]
        fn prop_model_mapping_correctness(model_name in model_name_strategy()) {
            let mapper = ModelMapper::empty();
            let result = mapper.map_model_name(&model_name);
            let lower_name = model_name.to_lowercase();

            // 精确匹配映射表 (与 default_mapping 中的一致)
            let exact_mapping: HashMap<&str, &str> = [
                ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5-thinking"),
                ("claude-3-5-sonnet-20241022", "claude-sonnet-4-5"),
                ("claude-3-5-sonnet-20240620", "claude-sonnet-4-5"),
                ("claude-opus-4", "claude-opus-4-5-thinking"),
                ("claude-opus-4-5-20251101", "claude-opus-4-5-thinking"),
                ("claude-opus-4-5", "claude-opus-4-5-thinking"),
                ("claude-haiku-4", "claude-sonnet-4-5"),
                ("claude-3-haiku-20240307", "claude-sonnet-4-5"),
                ("claude-haiku-4-5-20251001", "claude-sonnet-4-5"),
                ("gemini-3-pro-high", "gemini-3-pro-preview"),
                ("gemini-3-pro-low", "gemini-3-pro-preview"),
                ("gemini-3-flash", "gemini-3-flash-preview"),
            ].into_iter().collect();

            // 验证映射规则 (按优先级顺序)
            // 1. 已支持的模型直接透传
            if SUPPORTED_MODELS.contains(&model_name.as_str()) {
                prop_assert_eq!(result, model_name);
            }
            // 2. 精确匹配映射表
            else if let Some(expected) = exact_mapping.get(model_name.as_str()) {
                prop_assert_eq!(result, *expected);
            }
            // 3. 模糊匹配规则
            else if lower_name.contains("sonnet") || lower_name.contains("thinking") {
                prop_assert_eq!(result, "gemini-3-pro-preview");
            } else if lower_name.contains("haiku") {
                prop_assert_eq!(result, "gemini-2.0-flash-exp");
            } else if lower_name.contains("opus") {
                prop_assert_eq!(result, "gemini-3-pro-preview");
            } else if lower_name.starts_with("gemini-") {
                // 其他 Gemini 模型直接透传
                prop_assert_eq!(result, model_name);
            }
            // 其他情况回退到默认值 claude-sonnet-4-5，不做断言
        }

        /// **Property 11 (续): 用户自定义映射优先**
        /// **Validates: Requirements 6.5**
        #[test]
        fn prop_custom_mapping_priority(
            model_name in "[a-z]{5,15}",
            custom_target in "[a-z]{5,15}"
        ) {
            let mut custom = HashMap::new();
            custom.insert(model_name.clone(), custom_target.clone());

            let mapper = ModelMapper::new(custom);
            let result = mapper.map_model_name(&model_name);

            // 用户自定义映射应该优先
            prop_assert_eq!(result, custom_target);
        }

        /// **Property 12: web_search 模型强制**
        /// *For any* 请求，如果 tools 数组中包含 name 为 "web_search" 的工具，
        /// 则最终使用的模型应该是 "gemini-2.5-flash"。
        /// **Validates: Requirements 6.6**
        #[test]
        fn prop_web_search_forces_flash(
            model_name in model_name_strategy(),
            tools in tools_strategy()
        ) {
            let mapper = ModelMapper::empty();
            let result = mapper.map_model_with_tools(&model_name, tools.as_ref());

            let has_web_search = tools.as_ref()
                .map(|t| t.iter().any(|tool| {
                    tool.get("name")
                        .and_then(|n| n.as_str())
                        .map(|n| n == "web_search")
                        .unwrap_or(false)
                }))
                .unwrap_or(false);

            if has_web_search {
                // 有 web_search 工具时，必须返回 gemini-2.5-flash
                prop_assert_eq!(result, "gemini-2.5-flash");
            } else {
                // 无 web_search 时，应该返回正常映射结果
                let expected = mapper.map_model_name(&model_name);
                prop_assert_eq!(result, expected);
            }
        }
    }
}
