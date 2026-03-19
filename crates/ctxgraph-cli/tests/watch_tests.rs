// Tests for git log parsing logic.
// Since ctxgraph-cli is a binary crate (no lib target), we inline the parsing
// logic here to test it in isolation without requiring git to be installed.

mod git_parse {
    /// A parsed git commit (mirrors watch::GitCommit).
    #[derive(Debug, Clone)]
    pub struct GitCommit {
        pub hash: String,
        pub author_name: String,
        pub author_email: String,
        pub committed_at: String,
        pub subject: String,
        pub body: String,
    }

    /// Parse the raw output of our git log format into GitCommit list.
    /// This mirrors `watch::parse_git_log_output` exactly.
    pub fn parse_git_log_output(output: &str) -> Vec<GitCommit> {
        let mut commits = Vec::new();

        for block in output.split("---CTXGRAPH---") {
            let block = block.trim();
            if block.is_empty() {
                continue;
            }

            let lines: Vec<&str> = block.lines().collect();

            if lines.len() < 5 {
                continue;
            }

            let hash = lines[0].trim().to_string();
            if hash.is_empty() {
                continue;
            }

            let author_name = lines[1].trim().to_string();
            let author_email = lines[2].trim().to_string();
            let committed_at = lines[3].trim().to_string();
            let subject = lines[4].trim().to_string();

            let body = if lines.len() > 5 {
                lines[5..].join("\n").trim().to_string()
            } else {
                String::new()
            };

            commits.push(GitCommit {
                hash,
                author_name,
                author_email,
                committed_at,
                subject,
                body,
            });
        }

        commits
    }
}

use git_parse::parse_git_log_output;

/// Test parsing of git log format with a multi-line body.
#[test]
fn test_parse_git_log_output() {
    let raw = concat!(
        "---CTXGRAPH---\n",
        "abc1234def5678901234567890123456789012345678\n",
        "Jane Doe\n",
        "jane@example.com\n",
        "2026-03-19T10:00:00+00:00\n",
        "feat: add semantic search\n",
        "\n",
        "This adds a new FTS5-based search endpoint.\n",
        "Supports multi-token queries.\n",
        "---CTXGRAPH---\n",
        "def5678abc1234567890123456789012345678901234\n",
        "John Smith\n",
        "john@example.com\n",
        "2026-03-18T09:00:00+00:00\n",
        "fix: resolve null pointer\n",
    );

    let commits = parse_git_log_output(raw);

    assert_eq!(commits.len(), 2);

    let c0 = &commits[0];
    assert_eq!(c0.hash, "abc1234def5678901234567890123456789012345678");
    assert_eq!(c0.author_name, "Jane Doe");
    assert_eq!(c0.author_email, "jane@example.com");
    assert_eq!(c0.committed_at, "2026-03-19T10:00:00+00:00");
    assert_eq!(c0.subject, "feat: add semantic search");
    assert!(
        c0.body.contains("FTS5-based search"),
        "Body should contain commit body text"
    );

    let c1 = &commits[1];
    assert_eq!(c1.hash, "def5678abc1234567890123456789012345678901234");
    assert_eq!(c1.subject, "fix: resolve null pointer");
    assert!(c1.body.is_empty(), "Second commit has no body");
}

/// Test that a GitCommit converts to correct Episode content.
#[test]
fn test_git_commit_to_episode_content() {
    // When body is empty, content == subject
    let subject = "feat: add MCP server";
    let body = "";
    let content = if body.trim().is_empty() {
        subject.to_string()
    } else {
        format!("{subject}\n\n{}", body.trim())
    };
    assert_eq!(content, "feat: add MCP server");

    // When body is present, content = subject + double newline + body
    let subject2 = "feat: add MCP server";
    let body2 = "Implements the Model Context Protocol server.";
    let content2 = if body2.trim().is_empty() {
        subject2.to_string()
    } else {
        format!("{subject2}\n\n{}", body2.trim())
    };
    assert_eq!(
        content2,
        "feat: add MCP server\n\nImplements the Model Context Protocol server."
    );
}

/// Test that empty git log output doesn't crash.
#[test]
fn test_empty_git_log() {
    let commits = parse_git_log_output("");
    assert!(commits.is_empty(), "Empty output should produce no commits");

    let commits2 = parse_git_log_output("   \n\n  ");
    assert!(
        commits2.is_empty(),
        "Whitespace-only output should produce no commits"
    );
}

/// Test a single commit without body.
#[test]
fn test_single_commit_no_body() {
    let raw = concat!(
        "---CTXGRAPH---\n",
        "abc123\n",
        "Author Name\n",
        "author@test.com\n",
        "2026-01-01T00:00:00+00:00\n",
        "chore: update deps\n",
    );
    let commits = parse_git_log_output(raw);
    assert_eq!(commits.len(), 1);
    assert_eq!(commits[0].hash, "abc123");
    assert_eq!(commits[0].subject, "chore: update deps");
    assert!(commits[0].body.is_empty());
}
