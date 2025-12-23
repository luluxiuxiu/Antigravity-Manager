use std::sync::Arc;
use serde::{Deserialize, Serialize};
// use serde_json::Value;

// ===== OpenAI 格式定义 =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Array(Vec<ContentPart>),
}

impl MessageContent {
    /// 获取文本内容的预览
    pub fn preview(&self) -> String {
        match self {
            MessageContent::Text(s) => if s.chars().count() > 200 { format!("{}...", s.chars().take(200).collect::<String>()) } else { s.clone() },
            MessageContent::Array(parts) => {
                let mut s = String::new();
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        s.push_str(text);
                    }
                }
                if s.chars().count() > 200 { format!("{}...", s.chars().take(200).collect::<String>()) } else { s }
            }
        }
    }

    /// 获取完整文本内容
    pub fn text(&self) -> String {
         match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Array(parts) => {
                let mut s = String::new();
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        s.push_str(text);
                    }
                }
                s
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIMessage {
    pub role: String,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    #[serde(flatten)]
    pub extra: Option<std::collections::HashMap<String, serde_json::Value>>,
}

// ===== Anthropic 格式定义 =====

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: Option<String> },
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

/// 工具结果内容 - 支持字符串或数组格式
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Array(Vec<ToolResultBlock>),
}

/// 工具结果块
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<AnthropicImageSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub source_type: String, // "base64"
    pub media_type: String, // "image/jpeg", "image/png"
    pub data: String, // base64 string
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    // Anthropic content is always a list of blocks, but incoming JSON might process single string? 
    // Officially it can be string or array of blocks.
    #[serde(deserialize_with = "deserialize_anthropic_content")]
    pub content: Vec<AnthropicContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>, // 新增：用于存放思维链签名
}

// Custom deserializer to handle content being either string or array
fn deserialize_anthropic_content<'de, D>(deserializer: D) -> Result<Vec<AnthropicContent>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct ContentVisitor;

    impl<'de> serde::de::Visitor<'de> for ContentVisitor {
        type Value = Vec<AnthropicContent>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("string or list of content blocks")
        }

        fn visit_str<E>(self, value: &str) -> Result<Vec<AnthropicContent>, E>
        where
            E: serde::de::Error,
        {
            Ok(vec![AnthropicContent::Text { text: value.to_string() }])
        }

        fn visit_seq<V>(self, mut visitor: V) -> Result<Vec<AnthropicContent>, V::Error>
        where
            V: serde::de::SeqAccess<'de>,
        {
            let mut vec = Vec::new();
            while let Some(elem) = visitor.next_element()? {
                vec.push(elem);
            }
            Ok(vec)
        }
    }

    deserializer.deserialize_any(ContentVisitor)
}


/// Anthropic Thinking 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicThinking {
    #[serde(rename = "type")]
    pub thinking_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicChatRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default, deserialize_with = "deserialize_anthropic_system")]
    pub system: Option<String>, // System prompt is top-level, supports string or array
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinking>,
}

// Custom deserializer for system field (supports both string and array formats)
fn deserialize_anthropic_system<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Deserialize;
    
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum SystemField {
        String(String),
        Array(Vec<SystemBlock>),
    }
    
    #[derive(Deserialize)]
    struct SystemBlock {
        #[serde(rename = "type")]
        block_type: String,
        text: String,
    }
    
    let value = Option::<SystemField>::deserialize(deserializer)?;
    Ok(value.map(|v| match v {
        SystemField::String(s) => s,
        SystemField::Array(blocks) => {
            blocks.into_iter()
                .filter(|b| b.block_type == "text")
                .map(|b| b.text)
                .collect::<Vec<_>>()
                .join("\n")
        }
    }))
}

// ===== Gemini 格式定义 =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiInlineData {
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "inlineData")]
    pub inline_data: Option<GeminiInlineData>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "thoughtSignature")]
    pub thought_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "functionCall")]
    pub function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "functionResponse")]
    pub function_response: Option<FunctionResponse>,
}

/// Gemini 函数调用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// Gemini 函数响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponse {
    pub name: String,
    pub response: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// Anthropic 工具定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
}

/// Gemini 工具定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiTool {
    #[serde(skip_serializing_if = "Option::is_none", rename = "functionDeclarations")]
    pub function_declarations: Option<Vec<GeminiFunctionDeclaration>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "googleSearch")]
    pub google_search: Option<GoogleSearchConfig>,
}

/// Gemini 函数声明
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Google 搜索配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleSearchConfig {
    #[serde(skip_serializing_if = "Option::is_none", rename = "enhancedContent")]
    pub enhanced_content: Option<EnhancedContentConfig>,
}

/// 增强内容配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedContentConfig {
    #[serde(skip_serializing_if = "Option::is_none", rename = "imageSearch")]
    pub image_search: Option<ImageSearchConfig>,
}

/// 图片搜索配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSearchConfig {
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxResultCount")]
    pub max_result_count: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

impl GeminiPart {
    /// 创建文本 part
    pub fn text(text: String) -> Self {
        Self {
            text: Some(text),
            inline_data: None,
            thought_signature: None,
            thought: None,
            function_call: None,
            function_response: None,
        }
    }

    /// 创建图片 part
    pub fn image(inline_data: GeminiInlineData) -> Self {
        Self {
            text: None,
            inline_data: Some(inline_data),
            thought_signature: None,
            thought: None,
            function_call: None,
            function_response: None,
        }
    }

    /// 创建 thinking part
    pub fn thinking(text: String, signature: Option<String>) -> Self {
        Self {
            text: Some(text),
            inline_data: None,
            thought_signature: signature,
            thought: Some(true),
            function_call: None,
            function_response: None,
        }
    }

    /// 创建函数调用 part
    pub fn function_call_part(fc: FunctionCall, signature: Option<String>) -> Self {
        Self {
            text: None,
            inline_data: None,
            thought_signature: signature,
            thought: None,
            function_call: Some(fc),
            function_response: None,
        }
    }

    /// 创建函数响应 part
    pub fn function_response_part(fr: FunctionResponse) -> Self {
        Self {
            text: None,
            inline_data: None,
            thought_signature: None,
            thought: None,
            function_call: None,
            function_response: Some(fr),
        }
    }

    /// 创建仅包含签名的 part
    pub fn signature_only(signature: String) -> Self {
        Self {
            text: None,
            inline_data: None,
            thought_signature: Some(signature),
            thought: None,
            function_call: None,
            function_response: None,
        }
    }

    /// 创建空 part
    pub fn empty() -> Self {
        Self {
            text: Some(String::new()),
            inline_data: None,
            thought_signature: None,
            thought: None,
            function_call: None,
            function_response: None,
        }
    }
}



// ===== 格式转换 =====

/// 将 OpenAI messages 转换为 Gemini contents 数组（用于 Antigravity API）
pub fn convert_openai_to_gemini_contents(messages: &Vec<OpenAIMessage>) -> Vec<GeminiContent> {
    let mut contents = Vec::new();
    // 预编译正则，提取 markdown 图片：![alt](data:image/png;base64,.....)
    // 捕获组1: mime type, 捕获组2: base64 data (允许空格/换行)
    let re = regex::Regex::new(r"!\[.*?\]\(data:\s*(image/[a-zA-Z+-]+)\s*;\s*base64\s*,\s*([a-zA-Z0-9+/=\s]+)\)").unwrap();
    
    // 正则用于从 data URL 中提取 base64
    let re_data_url = regex::Regex::new(r"data:\s*(image/[a-zA-Z+-]+)\s*;\s*base64\s*,\s*([a-zA-Z0-9+/=\s]+)").unwrap();

    let mut pending_images: Vec<GeminiInlineData> = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        // Debug: 查看消息内容预览
        let preview = msg.content.preview();
        tracing::info!("Msg[{}][{}] content={:?}", i, msg.role, preview);

        // 角色映射
        let role = match msg.role.as_str() {
            "assistant" => "model",
            "system" => "user",
            _ => &msg.role,
        };
        
        let mut parts = Vec::new();
        
        // 1. 处理 Pending Images (Assist 历史图片注入到 User)
        if role == "user" && !pending_images.is_empty() {
             let count = pending_images.len();
             tracing::info!("向 User 消息注入 {} 张待处理图片 (上下文携带)", count);
             for img in pending_images.drain(..) {
                parts.push(GeminiPart::image(img));
            }
        }

        // 2. 解析当前消息内容 (支持 String 和 Array)
        match &msg.content {
            MessageContent::Text(text) => {
                // 处理 String 格式 (解析 Markdown 图片)
                let mut last_end = 0;
                for caps in re.captures_iter(text) {
                    let match_start = caps.get(0).map(|m| m.start()).unwrap_or(0);
                    let match_end = caps.get(0).map(|m| m.end()).unwrap_or(0);
                    
                    if match_start > last_end {
                        let text_part = &text[last_end..match_start];
                        if !text_part.is_empty() {
                            parts.push(GeminiPart::text(text_part.to_string()));
                        }
                    }
                    
                    let mime_type = caps.get(1).map(|m| m.as_str()).unwrap_or("image/png").to_string();
                    let data = caps.get(2).map(|m| m.as_str()).unwrap_or("").replace(|c: char| c.is_whitespace(), "");
                    let inline_data = GeminiInlineData { mime_type, data };

                    if role == "model" {
                        pending_images.push(inline_data);
                    } else {
                        parts.push(GeminiPart::image(inline_data));
                    }
                    last_end = match_end;
                }
                if last_end < text.len() {
                    let text_part = &text[last_end..];
                    if !text_part.is_empty() {
                        parts.push(GeminiPart::text(text_part.to_string()));
                    }
                }
            },
            MessageContent::Array(content_parts) => {
                // 处理 Array 格式 (多模态)
                for part in content_parts {
                    match part {
                        ContentPart::Text { text } => {
                            parts.push(GeminiPart::text(text.clone()));
                        },
                        ContentPart::ImageUrl { image_url } => {
                            let url = &image_url.url;
                            if let Some(caps) = re_data_url.captures(url) {
                                let mime_type = caps.get(1).map(|m| m.as_str()).unwrap_or("image/png").to_string();
                                let data = caps.get(2).map(|m| m.as_str()).unwrap_or("").replace(|c: char| c.is_whitespace(), "");
                                let inline_data = GeminiInlineData { mime_type: mime_type.clone(), data };
                                
                                if role == "model" {
                                    // 理论上 Model 消息不应该发这里，但防以后
                                    pending_images.push(inline_data);
                                } else {
                                    tracing::info!("解析到 Multimodal 图片数据 (Mime: {})", mime_type);
                                    parts.push(GeminiPart::image(inline_data));
                                }
                            } else {
                                tracing::warn!("忽略不支持的图片 URL 格式: {}", url);
                            }
                        }
                    }
                }
            }
        }
        
        // 3. 补全与清理
        if role == "model" && parts.is_empty() && !pending_images.is_empty() {
            parts.push(GeminiPart::text("[Image Generated]".to_string()));
        }

        if parts.is_empty() {
            parts.push(GeminiPart::empty());
        }
        
        contents.push(GeminiContent {
            role: role.to_string(),
            parts,
        });
    }
    
    // 合并连续 User 消息
    let mut i = 1;
    while i < contents.len() {
        if contents[i].role == "user" && contents[i-1].role == "user" {
            let mut parts_to_append = contents[i].parts.clone();
            
            let need_separator = if let Some(last_part) = contents[i-1].parts.last() {
                if let Some(first_part) = parts_to_append.first() {
                    last_part.text.is_some() && first_part.text.is_some()
                } else {
                    false
                }
            } else {
                false
            };
            
            if need_separator {
                contents[i-1].parts.push(GeminiPart::text("\n\n".to_string()));
            }
            
            contents[i-1].parts.append(&mut parts_to_append);
            contents.remove(i);
        } else {
            i += 1;
        }
    }
    
    contents
}

/// 将 Anthropic request 转换为 Gemini contents 数组
pub fn _convert_anthropic_to_gemini_contents(request: &AnthropicChatRequest) -> Vec<GeminiContent> {
    let mut contents = Vec::new();
    
    // 1. 处理 System Prompt
    // Gemini 将 System Prompt 视为 user 消息的一部分，或者放到 systemInstruction 中 (client.rs 处理 systemInstruction)
    // 这里我们仅处理 messages 部分。System prompt 将在 client.rs 中通过 systemInstruction 处理，
    // 或者如果需要兼容性，也可以在这里转为 User message。
    // ANTIGRAVITY 策略: System prompt 尽可能放到 systemInstruction。
    // 但是，Client 端的 convert 方法只接受 messages 向量，因此需要在 Client 中显式地把 request.system 拿出来。
    // converter 的这个函数只负责转换 messages 列表。

    for msg in &request.messages {
        let role = match msg.role.as_str() {
            "assistant" => "model",
            "user" => "user",
            _ => "user", // Default fallback
        };

        let mut parts = Vec::new();

        for content in &msg.content {
            match content {
                AnthropicContent::Text { text } => {
                    parts.push(GeminiPart::text(text.clone()));
                },
                AnthropicContent::Image { source } => {
                    // source_type: "base64", media_type: "image/jpeg", data: "..."
                    if source.source_type == "base64" {
                        parts.push(GeminiPart::image(GeminiInlineData {
                            mime_type: source.media_type.clone(),
                            data: source.data.clone(),
                        }));
                    }
                },
                AnthropicContent::Thinking { .. } => {
                    // Gemini 目前不支持直接接收 thinking 类型的输入块，忽略之以避免 400 错误
                    // 或者可以考虑将其转换为 text，但可能会干扰模型
                    tracing::debug!("Ignoring thinking block in input message");
                },
                AnthropicContent::ToolUse { .. } | AnthropicContent::ToolResult { .. } => {
                    // 这些类型在新的转换函数中处理
                    tracing::debug!("ToolUse/ToolResult should be handled by convert_anthropic_to_gemini_contents_v2");
                }
            }
        }

        contents.push(GeminiContent {
            role: role.to_string(),
            parts,
        });
    }

    // 合并连续 User 消息 (Gemini 不允许 consecutive user messages without model response)
    let mut i = 1;
    while i < contents.len() {
        if contents[i].role == contents[i-1].role {
             let mut parts_to_append = contents[i].parts.clone();
             contents[i-1].parts.append(&mut parts_to_append);
             contents.remove(i);
        } else {
            i += 1;
        }
    }

    contents
}

/// 支持思维链签名的 Anthropic -> Gemini 转换
pub async fn convert_anthropic_to_gemini_contents_ext(
    request: &AnthropicChatRequest,
    signature_map: Arc<tokio::sync::Mutex<std::collections::HashMap<String, String>>>
) -> Vec<GeminiContent> {
    let mut contents = Vec::new();
    let map = signature_map.lock().await;

    for msg in &request.messages {
        let role = match msg.role.as_str() {
            "assistant" => "model",
            "user" => "user",
            _ => "user",
        };

        let mut parts = Vec::new();

        for content in &msg.content {
            match content {
                AnthropicContent::Text { text } => {
                    parts.push(GeminiPart::text(text.clone()));
                },
                AnthropicContent::Image { source } => {
                    if source.source_type == "base64" {
                        parts.push(GeminiPart::image(GeminiInlineData {
                            mime_type: source.media_type.clone(),
                            data: source.data.clone(),
                        }));
                    }
                },
                AnthropicContent::Thinking { .. } => {
                    // 同样忽略 Thinking 输入
                    tracing::debug!("Ignoring thinking block in input message (ext)");
                },
                AnthropicContent::ToolUse { .. } | AnthropicContent::ToolResult { .. } => {
                    // 这些类型在新的转换函数中处理
                    tracing::debug!("ToolUse/ToolResult should be handled by convert_anthropic_to_gemini_contents_v2");
                }
            }
        }
        
        // 尝试回传 thoughtSignature
        // 如果是最后一轮的 assistant 消息，且我们暂存了签名，则尝试回填
        if role == "model" {
            let sig = map.get("latest").cloned();
            if let Some(s) = sig {
                parts.push(GeminiPart::signature_only(s));
            }
        }
        
        contents.push(GeminiContent {
            role: role.to_string(),
            parts,
        });
    }
    
    // 合并
    let mut i = 1;
    while i < contents.len() {
        if contents[i].role == contents[i-1].role {
             let mut parts_to_append = contents[i].parts.clone();
             contents[i-1].parts.append(&mut parts_to_append);
             contents.remove(i);
        } else {
            i += 1;
        }
    }

    contents
}

// ==================== tool_use/tool_result 转换函数 ====================

use crate::proxy::signature_manager::SignatureManager;

/// 将 Anthropic tool_use 转换为 Gemini functionCall part
/// 
/// # 参数
/// - `id`: tool_use 的 ID
/// - `name`: 工具名称
/// - `input`: 工具输入参数
/// - `signature`: 块内签名（优先使用）
/// - `signature_manager`: 签名管理器（用于从缓存恢复签名）
/// 
/// # 返回
/// - GeminiPart: 包含 functionCall 的 part
pub fn convert_tool_use_to_function_call(
    id: &str,
    name: &str,
    input: &serde_json::Value,
    signature: Option<&str>,
    signature_manager: Option<&SignatureManager>,
) -> GeminiPart {
    // 优先使用块内签名，否则从缓存恢复
    let thought_signature = if let Some(sig) = signature {
        Some(sig.to_string())
    } else if let Some(sm) = signature_manager {
        sm.get_tool_signature(id)
    } else {
        None
    };

    let fc = FunctionCall {
        name: name.to_string(),
        args: Some(input.clone()),
        id: Some(id.to_string()),
    };

    GeminiPart::function_call_part(fc, thought_signature)
}


/// 将 Anthropic tool_result 转换为 Gemini functionResponse part
/// 
/// # 参数
/// - `tool_use_id`: 对应的 tool_use ID
/// - `content`: 工具结果内容（字符串或数组）
/// - `tool_id_to_name`: tool_use.id -> name 的映射（用于还原函数名）
/// 
/// # 返回
/// - GeminiPart: 包含 functionResponse 的 part
pub fn convert_tool_result_to_function_response(
    tool_use_id: &str,
    content: &ToolResultContent,
    tool_id_to_name: Option<&std::collections::HashMap<String, String>>,
) -> GeminiPart {
    // 优先从映射中获取函数名，否则使用 tool_use_id 作为函数名
    let func_name = tool_id_to_name
        .and_then(|map| map.get(tool_use_id))
        .cloned()
        .unwrap_or_else(|| tool_use_id.to_string());

    // 处理 content 为字符串或数组的情况
    let result_content = match content {
        ToolResultContent::Text(text) => text.clone(),
        ToolResultContent::Array(blocks) => {
            // 将数组中的内容合并为字符串
            blocks
                .iter()
                .filter_map(|block| {
                    if block.block_type == "text" {
                        block.text.clone()
                    } else {
                        // 对于非文本类型，序列化为 JSON
                        serde_json::to_string(block).ok()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
    };

    let fr = FunctionResponse {
        name: func_name,
        response: serde_json::json!({ "result": result_content }),
        id: Some(tool_use_id.to_string()),
    };

    GeminiPart::function_response_part(fr)
}


// ==================== Schema 清理和转换函数 ====================

/// 需要从 Schema 中移除的字段
const FIELDS_TO_REMOVE: &[&str] = &["$schema", "additionalProperties", "format", "default", "uniqueItems"];

/// 验证字段（需要合并到 description 中）
const VALIDATION_FIELDS: &[&str] = &[
    "minLength", "maxLength", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems"
];

/// 清理 JSON Schema 以符合 Gemini 格式
/// 
/// 移除不支持的字段，将验证约束合并到 description 中
/// 
/// # 参数
/// - `schema`: 原始 JSON Schema
/// 
/// # 返回
/// - 清理后的 JSON Schema
pub fn clean_json_schema(schema: &serde_json::Value) -> serde_json::Value {
    match schema {
        serde_json::Value::Object(obj) => {
            let mut cleaned = serde_json::Map::new();
            let mut validations = Vec::new();

            // 收集验证约束
            for field in VALIDATION_FIELDS {
                if let Some(value) = obj.get(*field) {
                    validations.push(format!("{}: {}", field, value));
                }
            }

            for (key, value) in obj {
                // 跳过需要移除的字段
                if FIELDS_TO_REMOVE.contains(&key.as_str()) {
                    continue;
                }

                // 跳过验证字段（已收集到 validations）
                if VALIDATION_FIELDS.contains(&key.as_str()) {
                    continue;
                }

                // 处理 type 字段：将联合类型（如 ["string", "null"]）规范化为单一类型
                if key == "type" {
                    if let serde_json::Value::Array(types) = value {
                        let filtered: Vec<&serde_json::Value> = types
                            .iter()
                            .filter(|t| t.as_str() != Some("null"))
                            .collect();
                        let single_type = if let Some(first) = filtered.first() {
                            (*first).clone()
                        } else if let Some(first) = types.first() {
                            first.clone()
                        } else {
                            serde_json::Value::String("string".to_string())
                        };
                        cleaned.insert(key.clone(), single_type);
                        continue;
                    }
                }

                // 处理 description 字段：附加验证约束
                if key == "description" && !validations.is_empty() {
                    if let serde_json::Value::String(desc) = value {
                        let new_desc = format!("{} ({})", desc, validations.join(", "));
                        cleaned.insert(key.clone(), serde_json::Value::String(new_desc));
                        continue;
                    }
                }

                // 递归处理嵌套对象
                if value.is_object() || value.is_array() {
                    cleaned.insert(key.clone(), clean_json_schema(value));
                } else {
                    cleaned.insert(key.clone(), value.clone());
                }
            }

            // 如果有验证约束但没有 description，添加一个
            if !validations.is_empty() && !cleaned.contains_key("description") {
                cleaned.insert(
                    "description".to_string(),
                    serde_json::Value::String(format!("Validation: {}", validations.join(", ")))
                );
            }

            // 应用类型大写转换
            uppercase_schema_types(&serde_json::Value::Object(cleaned))
        },
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(clean_json_schema).collect())
        },
        _ => schema.clone(),
    }
}

/// 将 Schema 中的 type 字段值转换为大写
/// 
/// Gemini API 要求类型名称为大写（如 "STRING" 而非 "string"）
/// 
/// # 参数
/// - `schema`: JSON Schema
/// 
/// # 返回
/// - 类型大写后的 JSON Schema
pub fn uppercase_schema_types(schema: &serde_json::Value) -> serde_json::Value {
    match schema {
        serde_json::Value::Object(obj) => {
            let mut normalized = serde_json::Map::new();

            for (key, value) in obj {
                if key == "type" {
                    match value {
                        serde_json::Value::String(s) => {
                            normalized.insert(key.clone(), serde_json::Value::String(s.to_uppercase()));
                        },
                        serde_json::Value::Array(arr) => {
                            let uppercased: Vec<serde_json::Value> = arr
                                .iter()
                                .map(|item| {
                                    if let serde_json::Value::String(s) = item {
                                        serde_json::Value::String(s.to_uppercase())
                                    } else {
                                        item.clone()
                                    }
                                })
                                .collect();
                            normalized.insert(key.clone(), serde_json::Value::Array(uppercased));
                        },
                        _ => {
                            normalized.insert(key.clone(), value.clone());
                        }
                    }
                } else if value.is_object() || value.is_array() {
                    normalized.insert(key.clone(), uppercase_schema_types(value));
                } else {
                    normalized.insert(key.clone(), value.clone());
                }
            }

            serde_json::Value::Object(normalized)
        },
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(uppercase_schema_types).collect())
        },
        _ => schema.clone(),
    }
}

/// 将 Anthropic tools 定义转换为 Gemini functionDeclarations 格式
/// 
/// # 参数
/// - `tools`: Anthropic 工具定义列表
/// 
/// # 返回
/// - (GeminiTool 列表, 是否包含 web_search 工具)
pub fn convert_tools_to_function_declarations(tools: &[AnthropicTool]) -> (Vec<GeminiTool>, bool) {
    let has_web_search = tools.iter().any(|t| t.name == "web_search");

    if has_web_search {
        // 映射 web_search 到 googleSearch 工具
        let google_search_tool = GeminiTool {
            function_declarations: None,
            google_search: Some(GoogleSearchConfig {
                enhanced_content: Some(EnhancedContentConfig {
                    image_search: Some(ImageSearchConfig {
                        max_result_count: Some(5),
                    }),
                }),
            }),
        };
        return (vec![google_search_tool], true);
    }

    // 转换普通工具定义
    let function_declarations: Vec<GeminiFunctionDeclaration> = tools
        .iter()
        .filter_map(|tool| {
            tool.input_schema.as_ref().map(|schema| {
                GeminiFunctionDeclaration {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: Some(clean_json_schema(schema)),
                }
            })
        })
        .collect();

    if function_declarations.is_empty() {
        return (vec![], false);
    }

    let gemini_tool = GeminiTool {
        function_declarations: Some(function_declarations),
        google_search: None,
    };

    (vec![gemini_tool], false)
}


// ==================== 属性测试 ====================

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ==================== Property 6: tool_use 转换正确性 ====================
    // **Validates: Requirements 4.1, 4.2**
    // 
    // *For any* Anthropic tool_use 内容块，转换后的 Gemini functionCall 应该满足：
    // - functionCall.name == tool_use.name
    // - functionCall.args == tool_use.input
    // - functionCall.id == tool_use.id
    // - 如果 tool_use.signature 存在，则 part.thoughtSignature == tool_use.signature

    // 生成随机工具名称
    fn arb_tool_name() -> impl Strategy<Value = String> {
        "[a-z][a-z0-9_]{0,20}".prop_map(|s| s)
    }

    // 生成随机工具 ID
    fn arb_tool_id() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9]{8,16}".prop_map(|s| format!("toolu_{}", s))
    }

    // 生成随机签名
    fn arb_signature() -> impl Strategy<Value = Option<String>> {
        prop_oneof![
            Just(None),
            "[a-zA-Z0-9+/=]{20,50}".prop_map(|s| Some(s))
        ]
    }

    // 生成随机 JSON 输入
    fn arb_json_input() -> impl Strategy<Value = serde_json::Value> {
        prop_oneof![
            Just(serde_json::json!({})),
            Just(serde_json::json!({"key": "value"})),
            Just(serde_json::json!({"count": 42})),
            Just(serde_json::json!({"nested": {"a": 1, "b": "test"}})),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: anthropic-api-enhancement, Property 6: tool_use 转换正确性
        /// **Validates: Requirements 4.1, 4.2**
        #[test]
        fn test_tool_use_conversion_correctness(
            name in arb_tool_name(),
            id in arb_tool_id(),
            input in arb_json_input(),
            signature in arb_signature(),
        ) {
            // 执行转换
            let result = convert_tool_use_to_function_call(
                &id,
                &name,
                &input,
                signature.as_deref(),
                None, // 不使用签名管理器
            );

            // 验证 functionCall 存在
            prop_assert!(result.function_call.is_some(), "functionCall should exist");
            
            let fc = result.function_call.as_ref().expect("functionCall should exist");
            
            // 验证 name 相等
            prop_assert_eq!(&fc.name, &name, "functionCall.name should equal tool_use.name");
            
            // 验证 args 相等
            prop_assert_eq!(fc.args.as_ref(), Some(&input), "functionCall.args should equal tool_use.input");
            
            // 验证 id 相等
            prop_assert_eq!(fc.id.as_ref(), Some(&id), "functionCall.id should equal tool_use.id");
            
            // 验证签名处理
            if signature.is_some() {
                prop_assert_eq!(
                    result.thought_signature.as_ref(),
                    signature.as_ref(),
                    "thoughtSignature should equal tool_use.signature when present"
                );
            }
        }

        /// Feature: anthropic-api-enhancement, Property 6: tool_use 签名从缓存恢复
        /// **Validates: Requirements 4.3**
        #[test]
        fn test_tool_use_signature_recovery_from_cache(
            name in arb_tool_name(),
            id in arb_tool_id(),
            input in arb_json_input(),
            cached_signature in "[a-zA-Z0-9+/=]{20,50}",
        ) {
            // 创建签名管理器并存储签名
            let sm = SignatureManager::with_defaults();
            sm.store_tool_signature(&id, &cached_signature);

            // 执行转换（不提供块内签名）
            let result = convert_tool_use_to_function_call(
                &id,
                &name,
                &input,
                None, // 无块内签名
                Some(&sm), // 使用签名管理器
            );

            // 验证签名从缓存恢复
            prop_assert_eq!(
                result.thought_signature.as_ref(),
                Some(&cached_signature),
                "thoughtSignature should be recovered from cache"
            );
        }

        /// Feature: anthropic-api-enhancement, Property 6: 块内签名优先于缓存
        /// **Validates: Requirements 4.2, 4.3**
        #[test]
        fn test_tool_use_block_signature_priority(
            name in arb_tool_name(),
            id in arb_tool_id(),
            input in arb_json_input(),
            block_signature in "[a-zA-Z0-9+/=]{20,50}",
            cached_signature in "[a-zA-Z0-9+/=]{20,50}",
        ) {
            // 创建签名管理器并存储不同的签名
            let sm = SignatureManager::with_defaults();
            sm.store_tool_signature(&id, &cached_signature);

            // 执行转换（提供块内签名）
            let result = convert_tool_use_to_function_call(
                &id,
                &name,
                &input,
                Some(&block_signature), // 有块内签名
                Some(&sm), // 也有缓存签名
            );

            // 验证块内签名优先
            prop_assert_eq!(
                result.thought_signature.as_ref(),
                Some(&block_signature),
                "block signature should take priority over cached signature"
            );
        }
    }

    // ==================== Property 8: Schema 类型大写转换 ====================
    // **Validates: Requirements 4.6**
    //
    // *For any* JSON Schema 对象，经过 uppercase_schema_types 转换后，
    // 所有 "type" 字段的值应该是大写形式（如 "string" -> "STRING"）。

    // 生成随机 Schema 类型
    fn arb_schema_type() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("string".to_string()),
            Just("number".to_string()),
            Just("integer".to_string()),
            Just("boolean".to_string()),
            Just("array".to_string()),
            Just("object".to_string()),
        ]
    }

    // 生成简单的 JSON Schema
    fn arb_simple_schema() -> impl Strategy<Value = serde_json::Value> {
        arb_schema_type().prop_map(|t| {
            serde_json::json!({
                "type": t,
                "description": "A test field"
            })
        })
    }

    // 生成嵌套的 JSON Schema
    fn arb_nested_schema() -> impl Strategy<Value = serde_json::Value> {
        (arb_schema_type(), arb_schema_type()).prop_map(|(outer_type, inner_type)| {
            serde_json::json!({
                "type": outer_type,
                "properties": {
                    "nested": {
                        "type": inner_type,
                        "description": "Nested field"
                    }
                }
            })
        })
    }

    // 递归检查所有 type 字段是否为大写
    fn all_types_uppercase(value: &serde_json::Value) -> bool {
        match value {
            serde_json::Value::Object(obj) => {
                for (key, val) in obj {
                    if key == "type" {
                        match val {
                            serde_json::Value::String(s) => {
                                if s != &s.to_uppercase() {
                                    return false;
                                }
                            },
                            serde_json::Value::Array(arr) => {
                                for item in arr {
                                    if let serde_json::Value::String(s) = item {
                                        if s != &s.to_uppercase() {
                                            return false;
                                        }
                                    }
                                }
                            },
                            _ => {}
                        }
                    }
                    if !all_types_uppercase(val) {
                        return false;
                    }
                }
                true
            },
            serde_json::Value::Array(arr) => {
                arr.iter().all(all_types_uppercase)
            },
            _ => true,
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: anthropic-api-enhancement, Property 8: Schema 类型大写转换
        /// **Validates: Requirements 4.6**
        #[test]
        fn test_schema_type_uppercase_simple(schema in arb_simple_schema()) {
            let result = uppercase_schema_types(&schema);
            prop_assert!(
                all_types_uppercase(&result),
                "All type fields should be uppercase after conversion"
            );
        }

        /// Feature: anthropic-api-enhancement, Property 8: 嵌套 Schema 类型大写转换
        /// **Validates: Requirements 4.6**
        #[test]
        fn test_schema_type_uppercase_nested(schema in arb_nested_schema()) {
            let result = uppercase_schema_types(&schema);
            prop_assert!(
                all_types_uppercase(&result),
                "All nested type fields should be uppercase after conversion"
            );
        }

        /// Feature: anthropic-api-enhancement, Property 8: clean_json_schema 也应用大写转换
        /// **Validates: Requirements 4.6**
        #[test]
        fn test_clean_json_schema_applies_uppercase(schema in arb_simple_schema()) {
            let result = clean_json_schema(&schema);
            prop_assert!(
                all_types_uppercase(&result),
                "clean_json_schema should also apply uppercase conversion"
            );
        }

        /// Feature: anthropic-api-enhancement, Property 8: 联合类型数组大写转换
        /// **Validates: Requirements 4.6**
        #[test]
        fn test_schema_type_array_uppercase(
            type1 in arb_schema_type(),
            type2 in arb_schema_type(),
        ) {
            let schema = serde_json::json!({
                "type": [type1, type2]
            });
            let result = uppercase_schema_types(&schema);
            prop_assert!(
                all_types_uppercase(&result),
                "Array type values should all be uppercase"
            );
        }
    }
}