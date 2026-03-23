use std::sync::Arc;

use ctxgraph::Graph;
use ctxgraph_embed::EmbedEngine;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::protocol::{Request, Response, codes};
use crate::tools::{ToolContext, tool_result, tools_list};

pub struct McpServer {
    ctx: Arc<ToolContext>,
}

impl McpServer {
    pub fn new(graph: Graph, embed: Option<EmbedEngine>) -> Self {
        Self {
            ctx: Arc::new(ToolContext::new(graph, embed)),
        }
    }

    /// Run the MCP server: read JSON-RPC lines from stdin, write responses to stdout.
    pub async fn run(&self) {
        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();

        let mut reader = BufReader::new(stdin);
        let mut stdout = stdout;
        let mut line = String::new();

        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => break, // EOF
                Ok(_) => {}
                Err(e) => {
                    eprintln!("ctxgraph-mcp: read error: {e}");
                    break;
                }
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Parse JSON-RPC request
            let request: Request = match serde_json::from_str(trimmed) {
                Ok(r) => r,
                Err(e) => {
                    let resp = Response::error(
                        Value::Null,
                        codes::PARSE_ERROR,
                        &format!("parse error: {e}"),
                    );
                    Self::write_response(&mut stdout, &resp).await;
                    continue;
                }
            };

            // Notifications have no id — do not send a response
            if request.is_notification() {
                eprintln!("ctxgraph-mcp: notification: {}", request.method);
                continue;
            }

            let id = request.id.clone().unwrap_or(Value::Null);
            let response = self.dispatch(id.clone(), &request).await;
            Self::write_response(&mut stdout, &response).await;
        }
    }

    async fn dispatch(&self, id: Value, request: &Request) -> Response {
        match request.method.as_str() {
            "initialize" => {
                let result = json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "ctxgraph", "version": "0.3.0"}
                });
                Response::ok(id, result)
            }

            "tools/list" => Response::ok(id, tools_list()),

            "tools/call" => {
                let params = request.params.clone().unwrap_or(Value::Null);
                let tool_name = match params["name"].as_str() {
                    Some(n) => n.to_string(),
                    None => {
                        return Response::error(id, codes::INVALID_PARAMS, "missing tool name");
                    }
                };
                let args = params["arguments"].clone();

                let ctx = Arc::clone(&self.ctx);
                let result = match tool_name.as_str() {
                    "add_episode" => ctx.add_episode(args).await,
                    "search" => ctx.search(args).await,
                    "get_decision" => ctx.get_decision(args).await,
                    "traverse" => ctx.traverse(args).await,
                    "traverse_batch" => ctx.traverse_batch(args).await,
                    "find_precedents" => ctx.find_precedents(args).await,
                    other => Err(format!("unknown tool: {other}")),
                };

                Response::ok(id, tool_result(result))
            }

            "notifications/initialized" => {
                // Notification — should not reach here (filtered above), but handle gracefully
                Response::ok(id, Value::Null)
            }

            other => {
                eprintln!("ctxgraph-mcp: unknown method: {other}");
                Response::error(
                    id,
                    codes::METHOD_NOT_FOUND,
                    &format!("method not found: {other}"),
                )
            }
        }
    }

    async fn write_response(stdout: &mut tokio::io::Stdout, resp: &Response) {
        let mut line = match serde_json::to_string(resp) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("ctxgraph-mcp: failed to serialize response: {e}");
                return;
            }
        };
        line.push('\n');
        if let Err(e) = stdout.write_all(line.as_bytes()).await {
            eprintln!("ctxgraph-mcp: write error: {e}");
        }
        let _ = stdout.flush().await;
    }
}
