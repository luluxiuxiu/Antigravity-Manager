use serde::Serialize;
use serde_json::Value;

/// 流式状态机块类型
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BlockType {
    /// 无块
    None = 0,
    /// 文本块
    Text = 1,
    /// 思维链块
    Thinking = 2,
    /// 工具使用块
    ToolUse = 3,
}

/// SSE 事件结构
#[derive(Serialize, Debug, Clone)]
pub struct StreamEvent {
    /// 事件名称
    pub event: String,
    /// JSON 数据字符串
    pub data: String,
}

/// 增强版流式转换器
/// 
/// 将 Gemini 流式响应转换为 Anthropic SSE 格式
/// 支持 thinking、signature、tool_use 等场景
pub struct ClaudeStreamConverter {
    /// 当前块索引
    pub block_index: usize,
    /// 当前块类型
    current_type: BlockType,
    /// 是否已发送 message_start
    message_start_sent: bool,
    /// 是否已发送 message_stop
    message_stop_sent: bool,
    /// 是否使用了工具
    pub used_tool: bool,
    /// 暂存的签名（来自 thinking part）
    /// 在 thinking 块结束时发送 signature_delta
    pending_signature: Option<String>,
    /// 尾部签名（来自空 text part）
    /// 必须用独立的空 thinking 块承载
    trailing_signature: Option<String>,
    /// 是否有内容
    pub has_content: bool,
}

impl ClaudeStreamConverter {
    /// 创建新的流式转换器
    pub fn new() -> Self {
        Self {
            block_index: 0,
            current_type: BlockType::None,
            message_start_sent: false,
            message_stop_sent: false,
            used_tool: false,
            pending_signature: None,
            trailing_signature: None,
            has_content: false,
        }
    }

    /// 处理 Gemini chunk 并返回 Anthropic 事件列表
    /// 
    /// # 参数
    /// - `json_chunk`: Gemini 格式的 JSON chunk
    /// 
    /// # 返回
    /// - Vec<StreamEvent>: Anthropic SSE 事件列表
    pub fn process_chunk(&mut self, json_chunk: &Value) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        // 安全检查：空 choices
        let choices = match json_chunk.get("choices").and_then(|c| c.as_array()) {
            Some(arr) if !arr.is_empty() => arr,
            _ => return events,
        };

        let choice = &choices[0];
        let delta = &choice["delta"];
        
        let delta_content = delta.get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
            
        // 检查 thinking 字段
        let is_thought = delta.get("thought")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
            
        let thought_signature = delta.get("thoughtSignature")
            .and_then(|v| v.as_str());

        // 检查 function_call 字段
        let function_call = delta.get("functionCall");

        // 更新 has_content 状态
        if !delta_content.is_empty() || is_thought || thought_signature.is_some() || function_call.is_some() {
            self.has_content = true;
        }

        // --- 状态机逻辑 ---

        // 1. 处理 function_call（工具调用）
        if let Some(fc) = function_call {
            // 先处理 trailing_signature（来自之前的空 text）
            if self.trailing_signature.is_some() {
                events.extend(self.emit_trailing_signature_block());
            }
            
            events.extend(self.process_function_call(fc, thought_signature));
            return events;
        }

        // 2. 处理空 text 带签名（trailing signature）
        if delta_content.is_empty() && !is_thought && thought_signature.is_some() {
            self.trailing_signature = thought_signature.map(|s| s.to_string());
            return events;
        }

        // 3. 处理 Thinking（思维链）
        if is_thought {
            // 先处理 trailing_signature
            if self.trailing_signature.is_some() {
                events.extend(self.emit_trailing_signature_block());
            }
            
            events.extend(self.process_thinking(delta_content, thought_signature));
        } 
        // 4. 处理普通文本
        else if !delta_content.is_empty() {
            // 先处理 trailing_signature
            if self.trailing_signature.is_some() {
                events.extend(self.emit_trailing_signature_block());
            }
            
            // 非空 text 带签名时，需要特殊处理
            if thought_signature.is_some() {
                events.extend(self.process_text_with_signature(delta_content, thought_signature));
            } else {
                events.extend(self.process_text(delta_content));
            }
        }

        // 5. 处理 Stop Reason（如果存在）
        if let Some(reason_str) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            let usage = json_chunk.get("usage");
            events.extend(self.emit_finish(reason_str, usage));
        }

        events
    }

    /// 处理 thinking 内容
    /// 
    /// # 参数
    /// - `text`: thinking 文本内容
    /// - `signature`: 可选的签名
    /// 
    /// # 返回
    /// - Vec<StreamEvent>: 生成的事件列表
    fn process_thinking(&mut self, text: &str, signature: Option<&str>) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        // 关闭现有的 Text 块
        if self.current_type == BlockType::Text {
            events.extend(self.end_block());
        }

        // 开始 Thinking 块（如果未开始）
        if self.current_type == BlockType::None {
            events.push(self.create_event("content_block_start", serde_json::json!({
                "type": "content_block_start",
                "index": self.block_index,
                "content_block": { "type": "thinking", "thinking": "" }
            })));
            self.current_type = BlockType::Thinking;
        }

        // 发送 thinking_delta
        if !text.is_empty() {
            events.push(self.create_event("content_block_delta", serde_json::json!({
                "type": "content_block_delta",
                "index": self.block_index,
                "delta": { "type": "thinking_delta", "thinking": text }
            })));
        }

        // 暂存签名，在 thinking 块结束时发送
        if let Some(sig) = signature {
            self.pending_signature = Some(sig.to_string());
        }

        events
    }

    /// 处理普通文本
    /// 
    /// # 参数
    /// - `text`: 文本内容
    /// 
    /// # 返回
    /// - Vec<StreamEvent>: 生成的事件列表
    fn process_text(&mut self, text: &str) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        // 关闭现有的 Thinking 块
        if self.current_type == BlockType::Thinking {
            events.extend(self.end_block());
        }

        // 开始 Text 块（如果未开始）
        if self.current_type == BlockType::None {
            events.push(self.create_event("content_block_start", serde_json::json!({
                "type": "content_block_start",
                "index": self.block_index,
                "content_block": { "type": "text", "text": "" }
            })));
            self.current_type = BlockType::Text;
        }

        // 发送 text_delta
        events.push(self.create_event("content_block_delta", serde_json::json!({
            "type": "content_block_delta",
            "index": self.block_index,
            "delta": { "type": "text_delta", "text": text }
        })));

        events
    }

    /// 处理带签名的非空文本
    /// 
    /// 根据规范：非空 text 带签名必须立即处理，不能合并到当前 text 块
    /// 需要创建空 thinking 块承载签名（Claude 格式限制：text 不支持 signature）
    /// 
    /// # 参数
    /// - `text`: 文本内容
    /// - `signature`: 签名
    /// 
    /// # 返回
    /// - Vec<StreamEvent>: 生成的事件列表
    fn process_text_with_signature(&mut self, text: &str, signature: Option<&str>) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        // 1. 先关闭当前块
        events.extend(self.end_block());

        // 2. 开始新 text 块并发送内容
        events.push(self.create_event("content_block_start", serde_json::json!({
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": { "type": "text", "text": "" }
        })));
        self.current_type = BlockType::Text;

        events.push(self.create_event("content_block_delta", serde_json::json!({
            "type": "content_block_delta",
            "index": self.block_index,
            "delta": { "type": "text_delta", "text": text }
        })));

        // 3. 关闭 text 块
        events.extend(self.end_block());

        // 4. 创建空 thinking 块承载签名
        if let Some(sig) = signature {
            events.push(self.create_event("content_block_start", serde_json::json!({
                "type": "content_block_start",
                "index": self.block_index,
                "content_block": { "type": "thinking", "thinking": "" }
            })));

            events.push(self.create_event("content_block_delta", serde_json::json!({
                "type": "content_block_delta",
                "index": self.block_index,
                "delta": { "type": "thinking_delta", "thinking": "" }
            })));

            events.push(self.create_event("content_block_delta", serde_json::json!({
                "type": "content_block_delta",
                "index": self.block_index,
                "delta": { "type": "signature_delta", "signature": sig }
            })));

            events.push(self.create_event("content_block_stop", serde_json::json!({
                "type": "content_block_stop",
                "index": self.block_index
            })));
            self.block_index += 1;
        }

        events
    }

    /// 处理函数调用
    /// 
    /// # 参数
    /// - `fc`: functionCall JSON 对象
    /// - `signature`: 可选的签名
    /// 
    /// # 返回
    /// - Vec<StreamEvent>: 生成的事件列表
    fn process_function_call(&mut self, fc: &Value, signature: Option<&str>) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        // 关闭当前块
        events.extend(self.end_block());

        // 提取函数调用信息
        let name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("unknown");
        let args = fc.get("args").cloned().unwrap_or(serde_json::json!({}));
        let id = fc.get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{}-{}", name, uuid::Uuid::new_v4().to_string().split('-').next().unwrap_or("0")));

        // 构建 tool_use 块
        let mut content_block = serde_json::json!({
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": {}
        });

        // 如果有签名，附加到 tool_use 块
        if let Some(sig) = signature {
            content_block["signature"] = serde_json::Value::String(sig.to_string());
        }

        // 发送 content_block_start
        events.push(self.create_event("content_block_start", serde_json::json!({
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": content_block
        })));
        self.current_type = BlockType::ToolUse;

        // 发送 input_json_delta
        let args_str = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
        events.push(self.create_event("content_block_delta", serde_json::json!({
            "type": "content_block_delta",
            "index": self.block_index,
            "delta": { "type": "input_json_delta", "partial_json": args_str }
        })));

        self.used_tool = true;

        events
    }

    /// 发送 trailing_signature 的空 thinking 块
    /// 
    /// 根据官方文档：空 text 带签名必须用独立的空 thinking 块承载
    /// 
    /// # 返回
    /// - Vec<StreamEvent>: 生成的事件列表
    fn emit_trailing_signature_block(&mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        if let Some(sig) = self.trailing_signature.take() {
            // 先关闭当前块
            events.extend(self.end_block());

            // 创建空 thinking 块承载签名
            events.push(self.create_event("content_block_start", serde_json::json!({
                "type": "content_block_start",
                "index": self.block_index,
                "content_block": { "type": "thinking", "thinking": "" }
            })));

            events.push(self.create_event("content_block_delta", serde_json::json!({
                "type": "content_block_delta",
                "index": self.block_index,
                "delta": { "type": "thinking_delta", "thinking": "" }
            })));

            events.push(self.create_event("content_block_delta", serde_json::json!({
                "type": "content_block_delta",
                "index": self.block_index,
                "delta": { "type": "signature_delta", "signature": sig }
            })));

            events.push(self.create_event("content_block_stop", serde_json::json!({
                "type": "content_block_stop",
                "index": self.block_index
            })));
            self.block_index += 1;
        }

        events
    }

    /// 结束当前块
    /// 
    /// # 返回
    /// - Vec<StreamEvent>: 生成的事件列表
    fn end_block(&mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        if self.current_type == BlockType::None {
            return events;
        }

        // 如果是 thinking 块结束，先发送暂存的签名
        if self.current_type == BlockType::Thinking {
            if let Some(sig) = self.pending_signature.take() {
                events.push(self.create_event("content_block_delta", serde_json::json!({
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": { "type": "signature_delta", "signature": sig }
                })));
            }
        }

        events.push(self.create_event("content_block_stop", serde_json::json!({
            "type": "content_block_stop",
            "index": self.block_index
        })));
        self.block_index += 1;
        self.current_type = BlockType::None;

        events
    }

    /// 发送结束事件
    /// 
    /// # 参数
    /// - `finish_reason`: 结束原因
    /// - `usage`: 可选的使用量信息
    /// 
    /// # 返回
    /// - Vec<StreamEvent>: 生成的事件列表
    pub fn emit_finish(&mut self, finish_reason: &str, usage: Option<&Value>) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        // 关闭最后一个块
        events.extend(self.end_block());

        // 处理 trailing_signature（来自空 text part 的签名）
        if self.trailing_signature.is_some() {
            events.extend(self.emit_trailing_signature_block());
        }

        // 确定 stop_reason
        let stop_reason = if self.used_tool {
            "tool_use"
        } else {
            match finish_reason {
                "length" | "MAX_TOKENS" => "max_tokens",
                "stop" | "STOP" => "end_turn",
                "tool_calls" | "function_call" => "tool_use",
                _ => "end_turn"
            }
        };

        // 提取 usage 信息
        let output_tokens = usage
            .and_then(|u| u.get("completion_tokens").or(u.get("output_tokens")))
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        // 发送 message_delta
        events.push(self.create_event("message_delta", serde_json::json!({
            "type": "message_delta",
            "delta": { "stop_reason": stop_reason, "stop_sequence": null },
            "usage": { "output_tokens": output_tokens }
        })));

        // 发送 message_stop
        if !self.message_stop_sent {
            events.push(self.create_event("message_stop", serde_json::json!({
                "type": "message_stop"
            })));
            self.message_stop_sent = true;
        }

        events
    }

    /// 创建 message_start 事件
    /// 
    /// # 参数
    /// - `msg_id`: 消息 ID
    /// - `model`: 模型名称
    /// 
    /// # 返回
    /// - StreamEvent: message_start 事件
    pub fn create_message_start(msg_id: &str, model: &str) -> StreamEvent {
        let data = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": { "input_tokens": 0, "output_tokens": 0 }
            }
        });
        StreamEvent {
            event: "message_start".to_string(),
            data: data.to_string(),
        }
    }

    /// 创建 SSE 事件
    fn create_event(&self, event_name: &str, data: Value) -> StreamEvent {
        StreamEvent {
            event: event_name.to_string(),
            data: data.to_string(),
        }
    }

    /// 检查是否已发送 message_start
    pub fn is_message_start_sent(&self) -> bool {
        self.message_start_sent
    }

    /// 标记 message_start 已发送
    pub fn mark_message_start_sent(&mut self) {
        self.message_start_sent = true;
    }

    /// 获取当前块类型
    pub fn current_block_type(&self) -> BlockType {
        self.current_type
    }

    /// 检查是否有暂存的签名
    pub fn has_pending_signature(&self) -> bool {
        self.pending_signature.is_some()
    }

    /// 检查是否有 trailing 签名
    pub fn has_trailing_signature(&self) -> bool {
        self.trailing_signature.is_some()
    }
}

impl Default for ClaudeStreamConverter {
    fn default() -> Self {
        Self::new()
    }
}


// ==================== 属性测试 ====================

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ==================== Property 4: 流式响应事件顺序 ====================
    // **Validates: Requirements 3.1, 3.6**
    //
    // *For any* Gemini 流式响应，转换后的 Anthropic SSE 事件序列应该满足：
    // 1. 每个 content_block 必须以 content_block_start 开始，以 content_block_stop 结束
    // 2. 最后必须包含 message_delta 和 message_stop 事件

    // 生成随机文本内容
    fn arb_text_content() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9 ]{1,50}".prop_map(|s| s)
    }

    // 生成随机签名
    fn arb_signature() -> impl Strategy<Value = Option<String>> {
        prop_oneof![
            Just(None),
            "[a-zA-Z0-9+/=]{20,50}".prop_map(|s| Some(s))
        ]
    }

    // 生成随机 finish_reason
    fn arb_finish_reason() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("stop".to_string()),
            Just("STOP".to_string()),
            Just("length".to_string()),
            Just("MAX_TOKENS".to_string()),
            Just("tool_calls".to_string()),
        ]
    }

    // 生成 thinking chunk
    fn arb_thinking_chunk(text: String, signature: Option<String>) -> serde_json::Value {
        let mut delta = serde_json::json!({
            "content": text,
            "thought": true
        });
        if let Some(sig) = signature {
            delta["thoughtSignature"] = serde_json::Value::String(sig);
        }
        serde_json::json!({
            "choices": [{
                "delta": delta
            }]
        })
    }

    // 生成 text chunk
    fn arb_text_chunk(text: String) -> serde_json::Value {
        serde_json::json!({
            "choices": [{
                "delta": {
                    "content": text
                }
            }]
        })
    }

    // 生成 finish chunk
    fn arb_finish_chunk(reason: String) -> serde_json::Value {
        serde_json::json!({
            "choices": [{
                "delta": {},
                "finish_reason": reason
            }],
            "usage": {
                "completion_tokens": 100
            }
        })
    }

    // 检查事件序列是否符合规范
    fn validate_event_sequence(events: &[StreamEvent]) -> Result<(), String> {
        let mut block_started = false;
        let mut block_index = 0;
        let mut has_message_delta = false;
        let mut has_message_stop = false;

        for event in events {
            match event.event.as_str() {
                "content_block_start" => {
                    if block_started {
                        return Err("content_block_start without previous content_block_stop".to_string());
                    }
                    block_started = true;
                    
                    // 验证 index
                    let data: serde_json::Value = serde_json::from_str(&event.data)
                        .map_err(|e| format!("Invalid JSON: {}", e))?;
                    let idx = data.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    if idx != block_index {
                        return Err(format!("Expected block index {}, got {}", block_index, idx));
                    }
                },
                "content_block_stop" => {
                    if !block_started {
                        return Err("content_block_stop without content_block_start".to_string());
                    }
                    block_started = false;
                    block_index += 1;
                },
                "content_block_delta" => {
                    if !block_started {
                        return Err("content_block_delta without content_block_start".to_string());
                    }
                },
                "message_delta" => {
                    has_message_delta = true;
                },
                "message_stop" => {
                    has_message_stop = true;
                },
                _ => {}
            }
        }

        // 检查是否有未关闭的块
        if block_started {
            return Err("Unclosed content block".to_string());
        }

        // 检查是否有 message_delta 和 message_stop
        if !has_message_delta {
            return Err("Missing message_delta event".to_string());
        }
        if !has_message_stop {
            return Err("Missing message_stop event".to_string());
        }

        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: anthropic-api-enhancement, Property 4: 流式响应事件顺序
        /// **Validates: Requirements 3.1, 3.6**
        #[test]
        fn test_stream_event_sequence_thinking(
            text in arb_text_content(),
            signature in arb_signature(),
            finish_reason in arb_finish_reason(),
        ) {
            let mut converter = ClaudeStreamConverter::new();
            let mut all_events = Vec::new();

            // 处理 thinking chunk
            let chunk = arb_thinking_chunk(text, signature);
            all_events.extend(converter.process_chunk(&chunk));

            // 处理 finish chunk
            let finish_chunk = arb_finish_chunk(finish_reason);
            all_events.extend(converter.process_chunk(&finish_chunk));

            // 验证事件序列
            prop_assert!(
                validate_event_sequence(&all_events).is_ok(),
                "Event sequence validation failed: {:?}",
                validate_event_sequence(&all_events)
            );
        }

        /// Feature: anthropic-api-enhancement, Property 4: 流式响应事件顺序（文本）
        /// **Validates: Requirements 3.1, 3.6**
        #[test]
        fn test_stream_event_sequence_text(
            text in arb_text_content(),
            finish_reason in arb_finish_reason(),
        ) {
            let mut converter = ClaudeStreamConverter::new();
            let mut all_events = Vec::new();

            // 处理 text chunk
            let chunk = arb_text_chunk(text);
            all_events.extend(converter.process_chunk(&chunk));

            // 处理 finish chunk
            let finish_chunk = arb_finish_chunk(finish_reason);
            all_events.extend(converter.process_chunk(&finish_chunk));

            // 验证事件序列
            prop_assert!(
                validate_event_sequence(&all_events).is_ok(),
                "Event sequence validation failed: {:?}",
                validate_event_sequence(&all_events)
            );
        }

        /// Feature: anthropic-api-enhancement, Property 4: 流式响应事件顺序（混合）
        /// **Validates: Requirements 3.1, 3.6**
        #[test]
        fn test_stream_event_sequence_mixed(
            thinking_text in arb_text_content(),
            text in arb_text_content(),
            signature in arb_signature(),
            finish_reason in arb_finish_reason(),
        ) {
            let mut converter = ClaudeStreamConverter::new();
            let mut all_events = Vec::new();

            // 处理 thinking chunk
            let thinking_chunk = arb_thinking_chunk(thinking_text, signature);
            all_events.extend(converter.process_chunk(&thinking_chunk));

            // 处理 text chunk
            let text_chunk = arb_text_chunk(text);
            all_events.extend(converter.process_chunk(&text_chunk));

            // 处理 finish chunk
            let finish_chunk = arb_finish_chunk(finish_reason);
            all_events.extend(converter.process_chunk(&finish_chunk));

            // 验证事件序列
            prop_assert!(
                validate_event_sequence(&all_events).is_ok(),
                "Event sequence validation failed: {:?}",
                validate_event_sequence(&all_events)
            );
        }
    }

    // ==================== Property 5: 内容块类型正确性 ====================
    // **Validates: Requirements 3.2, 3.4, 3.5**
    //
    // *For any* Gemini 响应 part：
    // - 如果 part.thought == true，应该生成 thinking 类型的 content_block
    // - 如果 part.text 存在且 part.thought != true，应该生成 text 类型的 content_block
    // - 如果 part.functionCall 存在，应该生成 tool_use 类型的 content_block

    // 生成随机函数名
    fn arb_function_name() -> impl Strategy<Value = String> {
        "[a-z][a-z0-9_]{0,20}".prop_map(|s| s)
    }

    // 生成 function_call chunk
    fn arb_function_call_chunk(name: String, signature: Option<String>) -> serde_json::Value {
        let mut delta = serde_json::json!({
            "functionCall": {
                "name": name,
                "args": {"key": "value"},
                "id": format!("call_{}", name)
            }
        });
        if let Some(sig) = signature {
            delta["thoughtSignature"] = serde_json::Value::String(sig);
        }
        serde_json::json!({
            "choices": [{
                "delta": delta
            }]
        })
    }

    // 从事件中提取 content_block 类型
    fn extract_block_type(events: &[StreamEvent]) -> Option<String> {
        for event in events {
            if event.event == "content_block_start" {
                let data: serde_json::Value = serde_json::from_str(&event.data).ok()?;
                let block_type = data
                    .get("content_block")
                    .and_then(|cb| cb.get("type"))
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string());
                return block_type;
            }
        }
        None
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: anthropic-api-enhancement, Property 5: thinking 块类型正确性
        /// **Validates: Requirements 3.2**
        #[test]
        fn test_thinking_block_type(
            text in arb_text_content(),
            signature in arb_signature(),
        ) {
            let mut converter = ClaudeStreamConverter::new();
            
            // 处理 thinking chunk
            let chunk = arb_thinking_chunk(text, signature);
            let events = converter.process_chunk(&chunk);

            // 验证生成的是 thinking 类型的块
            let block_type = extract_block_type(&events);
            prop_assert_eq!(
                block_type,
                Some("thinking".to_string()),
                "Expected thinking block type"
            );
        }

        /// Feature: anthropic-api-enhancement, Property 5: text 块类型正确性
        /// **Validates: Requirements 3.4**
        #[test]
        fn test_text_block_type(text in arb_text_content()) {
            let mut converter = ClaudeStreamConverter::new();
            
            // 处理 text chunk
            let chunk = arb_text_chunk(text);
            let events = converter.process_chunk(&chunk);

            // 验证生成的是 text 类型的块
            let block_type = extract_block_type(&events);
            prop_assert_eq!(
                block_type,
                Some("text".to_string()),
                "Expected text block type"
            );
        }

        /// Feature: anthropic-api-enhancement, Property 5: tool_use 块类型正确性
        /// **Validates: Requirements 3.5**
        #[test]
        fn test_tool_use_block_type(
            name in arb_function_name(),
            signature in arb_signature(),
        ) {
            let mut converter = ClaudeStreamConverter::new();
            
            // 处理 function_call chunk
            let chunk = arb_function_call_chunk(name, signature);
            let events = converter.process_chunk(&chunk);

            // 验证生成的是 tool_use 类型的块
            let block_type = extract_block_type(&events);
            prop_assert_eq!(
                block_type,
                Some("tool_use".to_string()),
                "Expected tool_use block type"
            );
        }

        /// Feature: anthropic-api-enhancement, Property 5: stop_reason 正确性
        /// **Validates: Requirements 3.6**
        #[test]
        fn test_stop_reason_correctness(finish_reason in arb_finish_reason()) {
            let mut converter = ClaudeStreamConverter::new();
            
            // 处理 text chunk
            let text_chunk = arb_text_chunk("test".to_string());
            converter.process_chunk(&text_chunk);

            // 处理 finish chunk
            let finish_chunk = arb_finish_chunk(finish_reason.clone());
            let events = converter.process_chunk(&finish_chunk);

            // 找到 message_delta 事件并验证 stop_reason
            let message_delta = events.iter().find(|e| e.event == "message_delta");
            prop_assert!(message_delta.is_some(), "Should have message_delta event");

            let data: serde_json::Value = serde_json::from_str(&message_delta.expect("checked above").data)
                .expect("Valid JSON");
            let stop_reason = data
                .get("delta")
                .and_then(|d| d.get("stop_reason"))
                .and_then(|r| r.as_str())
                .expect("stop_reason should exist");

            // 验证 stop_reason 映射正确
            let expected = match finish_reason.as_str() {
                "length" | "MAX_TOKENS" => "max_tokens",
                "stop" | "STOP" => "end_turn",
                "tool_calls" | "function_call" => "tool_use",
                _ => "end_turn"
            };
            prop_assert_eq!(stop_reason, expected, "stop_reason should be correctly mapped");
        }

        /// Feature: anthropic-api-enhancement, Property 5: tool_use 设置 used_tool 标志
        /// **Validates: Requirements 3.5, 3.6**
        #[test]
        fn test_tool_use_sets_used_tool_flag(
            name in arb_function_name(),
            signature in arb_signature(),
        ) {
            let mut converter = ClaudeStreamConverter::new();
            
            // 处理 function_call chunk
            let chunk = arb_function_call_chunk(name, signature);
            converter.process_chunk(&chunk);

            // 验证 used_tool 标志被设置
            prop_assert!(converter.used_tool, "used_tool flag should be set after function_call");

            // 处理 finish chunk
            let finish_chunk = arb_finish_chunk("stop".to_string());
            let events = converter.process_chunk(&finish_chunk);

            // 验证 stop_reason 是 tool_use
            let message_delta = events.iter().find(|e| e.event == "message_delta");
            prop_assert!(message_delta.is_some(), "Should have message_delta event");

            let data: serde_json::Value = serde_json::from_str(&message_delta.expect("checked above").data)
                .expect("Valid JSON");
            let stop_reason = data
                .get("delta")
                .and_then(|d| d.get("stop_reason"))
                .and_then(|r| r.as_str())
                .expect("stop_reason should exist");

            prop_assert_eq!(stop_reason, "tool_use", "stop_reason should be tool_use when used_tool is true");
        }
    }

    // ==================== 额外测试：signature_delta 事件 ====================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Feature: anthropic-api-enhancement, Property 4: signature_delta 在 thinking 块结束时发送
        /// **Validates: Requirements 3.3**
        #[test]
        fn test_signature_delta_on_thinking_end(
            text in arb_text_content(),
            signature in "[a-zA-Z0-9+/=]{20,50}",
        ) {
            let mut converter = ClaudeStreamConverter::new();
            let mut all_events = Vec::new();

            // 处理带签名的 thinking chunk
            let chunk = arb_thinking_chunk(text, Some(signature.clone()));
            all_events.extend(converter.process_chunk(&chunk));

            // 处理 finish chunk（会触发 end_block）
            let finish_chunk = arb_finish_chunk("stop".to_string());
            all_events.extend(converter.process_chunk(&finish_chunk));

            // 验证有 signature_delta 事件
            let has_signature_delta = all_events.iter().any(|e| {
                if e.event == "content_block_delta" {
                    if let Ok(data) = serde_json::from_str::<serde_json::Value>(&e.data) {
                        return data
                            .get("delta")
                            .and_then(|d| d.get("type"))
                            .and_then(|t| t.as_str())
                            == Some("signature_delta");
                    }
                }
                false
            });

            prop_assert!(has_signature_delta, "Should have signature_delta event when thinking has signature");
        }

        /// Feature: anthropic-api-enhancement, Property 4: trailing_signature 生成空 thinking 块
        /// **Validates: Requirements 3.7**
        #[test]
        fn test_trailing_signature_creates_empty_thinking_block(
            signature in "[a-zA-Z0-9+/=]{20,50}",
        ) {
            let mut converter = ClaudeStreamConverter::new();
            let mut all_events = Vec::new();

            // 处理空 text 带签名的 chunk（trailing signature）
            let chunk = serde_json::json!({
                "choices": [{
                    "delta": {
                        "content": "",
                        "thoughtSignature": signature
                    }
                }]
            });
            all_events.extend(converter.process_chunk(&chunk));

            // 处理 finish chunk
            let finish_chunk = arb_finish_chunk("stop".to_string());
            all_events.extend(converter.process_chunk(&finish_chunk));

            // 验证有空 thinking 块承载签名
            let mut found_empty_thinking = false;
            let mut found_signature_delta = false;

            for event in &all_events {
                if event.event == "content_block_start" {
                    if let Ok(data) = serde_json::from_str::<serde_json::Value>(&event.data) {
                        let block_type = data
                            .get("content_block")
                            .and_then(|cb| cb.get("type"))
                            .and_then(|t| t.as_str());
                        let thinking_content = data
                            .get("content_block")
                            .and_then(|cb| cb.get("thinking"))
                            .and_then(|t| t.as_str());
                        
                        if block_type == Some("thinking") && thinking_content == Some("") {
                            found_empty_thinking = true;
                        }
                    }
                }
                if event.event == "content_block_delta" {
                    if let Ok(data) = serde_json::from_str::<serde_json::Value>(&event.data) {
                        if data
                            .get("delta")
                            .and_then(|d| d.get("type"))
                            .and_then(|t| t.as_str())
                            == Some("signature_delta")
                        {
                            found_signature_delta = true;
                        }
                    }
                }
            }

            prop_assert!(
                found_empty_thinking && found_signature_delta,
                "Should have empty thinking block with signature_delta for trailing signature"
            );
        }
    }
}
