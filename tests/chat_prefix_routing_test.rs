//! lingjing patch: cache-aware routing text for chat requests without a session id.
//!
//! The property that matters is prefix alignment: the routing text of turn N+1
//! must start with the routing text of turn N, so the cache-aware radix tree keeps
//! a conversation on the replica that already holds its KV prefix.

use serde_json::json;
use vllm_router_rs::protocols::spec::{ChatCompletionRequest, GenerationRequest};

fn req(body: serde_json::Value) -> ChatCompletionRequest {
    serde_json::from_value(body).expect("valid chat request")
}

fn tools() -> serde_json::Value {
    json!([
        {"type": "function", "function": {"name": "search", "description": "web search",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "calc", "parameters": {"type": "object"}}}
    ])
}

#[test]
fn follow_up_turn_extends_previous_turn() {
    let turn1 = req(json!({"model": "m", "tools": tools(), "messages": [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "查一下天气"}
    ]}));
    let turn2 = req(json!({"model": "m", "tools": tools(), "messages": [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "查一下天气"},
        {"role": "assistant", "content": null, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{\"q\":\"天气\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "晴 25 度"},
        {"role": "user", "content": [{"type": "text", "text": "那明天呢"}, {"type": "image_url", "image_url": {"url": "data:,"}}]}
    ]}));
    let t1 = turn1.extract_text_for_routing();
    let t2 = turn2.extract_text_for_routing();
    assert!(!t1.is_empty());
    assert!(t2.starts_with(&t1), "turn 2 must extend turn 1:\nturn1={t1}\nturn2={t2}");
    assert!(t2.contains("assistant_tool_call:search:{\"q\":\"天气\"}"));
    assert!(t2.contains("tool:晴 25 度"));
    assert!(t2.ends_with("user:那明天呢"));
}

#[test]
fn tools_come_first_in_wire_order() {
    let r = req(json!({"model": "m", "tools": tools(), "messages": [{"role": "user", "content": "hi"}]}));
    let text = r.extract_text_for_routing();
    let search = text.find("tool_schema:search:").expect("search tool");
    let calc = text.find("tool_schema:calc:").expect("calc tool");
    let user = text.find("user:hi").expect("user message");
    assert!(search < calc && calc < user, "tools must precede messages in wire order: {text}");
}

#[test]
fn different_tools_do_not_look_like_the_same_prefix() {
    let a = req(json!({"model": "m", "tools": tools(), "messages": [{"role": "user", "content": "hi"}]}));
    let b = req(json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}));
    assert_ne!(a.extract_text_for_routing(), b.extract_text_for_routing());
    assert!(!a.extract_text_for_routing().starts_with(&b.extract_text_for_routing()));
}

#[test]
fn session_id_still_wins() {
    let r = req(json!({"model": "m", "session_params": {"session_id": "s-9"},
        "messages": [{"role": "user", "content": "hi"}]}));
    assert_eq!(r.extract_text_for_routing(), "s-9");
    let blank = req(json!({"model": "m", "session_params": {"session_id": "  "},
        "messages": [{"role": "user", "content": "hi"}]}));
    assert_eq!(blank.extract_text_for_routing(), "user:hi");
}

#[test]
fn whitespace_is_preserved_for_prefix_fidelity() {
    // Engines cache on exact tokens; trimming would make " a" and "a" look identical.
    let a = req(json!({"model": "m", "messages": [{"role": "user", "content": " a"}]}));
    let b = req(json!({"model": "m", "messages": [{"role": "user", "content": "a"}]}));
    assert_ne!(a.extract_text_for_routing(), b.extract_text_for_routing());
}
