use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A JSON-RPC 2.0 request message.
#[derive(Debug, Deserialize)]
pub struct Request {
    #[allow(dead_code)]
    pub jsonrpc: String,
    /// `null` for notifications (no response expected).
    pub id: Option<Value>,
    pub method: String,
    pub params: Option<Value>,
}

impl Request {
    /// Returns true if this is a notification (id is absent / null).
    pub fn is_notification(&self) -> bool {
        self.id.is_none()
    }
}

/// A JSON-RPC 2.0 response message.
#[derive(Debug, Serialize)]
pub struct Response {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

/// A JSON-RPC error object.
#[derive(Debug, Serialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
}

impl Response {
    pub fn ok(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Value, code: i64, message: &str) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(RpcError {
                code,
                message: message.to_string(),
            }),
        }
    }
}

/// Standard JSON-RPC error codes.
pub mod codes {
    pub const PARSE_ERROR: i64 = -32700;
    #[allow(dead_code)]
    pub const INVALID_REQUEST: i64 = -32600;
    pub const METHOD_NOT_FOUND: i64 = -32601;
    pub const INVALID_PARAMS: i64 = -32602;
    #[allow(dead_code)]
    pub const INTERNAL_ERROR: i64 = -32603;
}
