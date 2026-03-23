use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

struct McpProcess {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
}

impl McpProcess {
    fn spawn(db_path: &str) -> Self {
        let binary = env!("CARGO_BIN_EXE_ctxgraph-mcp");
        let mut child = Command::new(binary)
            .arg("--db")
            .arg(db_path)
            .env("CTXGRAPH_NO_EMBED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn ctxgraph-mcp binary");

        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        Self {
            child,
            stdin,
            reader: BufReader::new(stdout),
        }
    }

    fn send(&mut self, msg: Value) {
        let line = serde_json::to_string(&msg).unwrap() + "\n";
        self.stdin.write_all(line.as_bytes()).unwrap();
        self.stdin.flush().unwrap();
    }

    fn recv(&mut self) -> Value {
        let mut line = String::new();
        self.reader.read_line(&mut line).unwrap();
        serde_json::from_str(line.trim()).expect("invalid JSON response")
    }
}

impl Drop for McpProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

#[test]
fn test_mcp_initialize_handshake() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("test.db").to_str().unwrap().to_string();

    let mut mcp = McpProcess::spawn(&db);

    mcp.send(json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "test", "version": "0.1"}
        }
    }));

    let resp = mcp.recv();
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);
    assert!(
        resp["result"]["serverInfo"]["name"]
            .as_str()
            .unwrap()
            .contains("ctxgraph")
    );
    assert!(resp["result"]["capabilities"]["tools"].is_object());
}

#[test]
fn test_mcp_tools_list() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("test.db").to_str().unwrap().to_string();
    let mut mcp = McpProcess::spawn(&db);

    // initialize first
    mcp.send(json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}));
    mcp.recv(); // consume response

    mcp.send(json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}));
    let resp = mcp.recv();

    let tools = resp["result"]["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"add_episode"));
    assert!(names.contains(&"search"));
    assert!(names.contains(&"get_decision"));
    assert!(names.contains(&"traverse"));
    assert!(names.contains(&"find_precedents"));
}

#[test]
fn test_mcp_add_episode_and_search() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("test.db").to_str().unwrap().to_string();
    let mut mcp = McpProcess::spawn(&db);

    mcp.send(json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}));
    mcp.recv();

    // Add episode
    mcp.send(json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "add_episode",
            "arguments": {"text": "Chose PostgreSQL over MySQL for the billing service"}
        }
    }));
    let add_resp = mcp.recv();
    assert_eq!(add_resp["id"], 2);
    let content = &add_resp["result"]["content"][0]["text"];
    let result: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
    assert!(result["episode_id"].is_string());

    // Search for it
    mcp.send(json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "search",
            "arguments": {"query": "PostgreSQL billing", "limit": 5}
        }
    }));
    let search_resp = mcp.recv();
    let search_content = &search_resp["result"]["content"][0]["text"];
    let results: Value = serde_json::from_str(search_content.as_str().unwrap()).unwrap();
    // Should find at least one result
    assert!(
        results.as_array().map(|a| !a.is_empty()).unwrap_or(false),
        "search should return results"
    );
}

#[test]
fn test_mcp_unknown_method_returns_error() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("test.db").to_str().unwrap().to_string();
    let mut mcp = McpProcess::spawn(&db);

    mcp.send(json!({"jsonrpc":"2.0","id":1,"method":"nonexistent","params":{}}));
    let resp = mcp.recv();
    assert!(resp["error"].is_object());
    assert_eq!(resp["error"]["code"], -32601);
}
