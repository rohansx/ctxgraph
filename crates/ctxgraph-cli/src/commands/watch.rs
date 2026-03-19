use std::path::{Path, PathBuf};
use std::process::Command;

use colored::Colorize;
use ctxgraph::{Episode, Graph};

use super::open_graph;

/// A parsed git commit.
#[derive(Debug, Clone)]
pub struct GitCommit {
    pub hash: String,
    pub author_name: String,
    pub author_email: String,
    pub committed_at: String, // ISO 8601
    pub subject: String,
    pub body: String,
}

/// Parse the raw output of our git log format into GitCommit list.
pub(crate) fn parse_git_log_output(output: &str) -> Vec<GitCommit> {
    let mut commits = Vec::new();

    // Split on our sentinel — the first element before any sentinel will be empty or whitespace
    for block in output.split("---CTXGRAPH---") {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }

        let lines: Vec<&str> = block.lines().collect();

        // Need at least 5 lines: hash, author_name, author_email, committed_at, subject
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

        // Body is everything from line 5 onward (may be empty)
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

/// Run `git log` and parse commits.
pub fn get_git_commits(
    repo: &Path,
    last: usize,
    since: Option<&str>,
) -> Result<Vec<GitCommit>, String> {
    let format = "--format=---CTXGRAPH---%n%H%n%an%n%ae%n%ai%n%s%n%b";

    let mut cmd = Command::new("git");
    cmd.current_dir(repo)
        .arg("log")
        .arg(format)
        .arg(format!("-n{last}"));

    if let Some(since_date) = since {
        cmd.arg(format!("--since={since_date}"));
    }

    let output = cmd.output().map_err(|e| {
        format!("Failed to run git: {e}. Is git installed?")
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "git log failed: {stderr}. Is this a git repository?"
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(parse_git_log_output(&stdout))
}

/// Import git commits into the graph.
pub fn run_watch(
    graph: &Graph,
    last: usize,
    since: Option<&str>,
    repo: Option<&PathBuf>,
    install_hook: bool,
) -> Result<(), String> {
    let default_repo = std::env::current_dir()
        .map_err(|e| format!("Cannot determine current directory: {e}"))?;
    let repo_path = repo.map(|p| p.as_path()).unwrap_or(default_repo.as_path());

    if install_hook {
        install_post_commit_hook(repo_path)?;
    }

    println!("Scanning git log (last {} commits)...", last);

    let commits = get_git_commits(repo_path, last, since)
        .map_err(|e| e)?;

    let mut imported = 0usize;
    let mut skipped = 0usize;

    for commit in &commits {
        // Check if already ingested
        let already_exists = graph
            .has_episode_by_git_hash(&commit.hash)
            .map_err(|e| format!("Database error: {e}"))?;

        let short_hash = &commit.hash[..commit.hash.len().min(7)];

        if already_exists {
            println!(
                "  {} {} \"{}\" — already ingested, skipping",
                "→".yellow(),
                short_hash,
                truncate(&commit.subject, 60)
            );
            skipped += 1;
            continue;
        }

        // Build episode content
        let content = if commit.body.trim().is_empty() {
            commit.subject.clone()
        } else {
            format!("{}\n\n{}", commit.subject, commit.body.trim())
        };

        let episode = Episode::builder(&content)
            .source("git")
            .meta("commit_hash", commit.hash.as_str())
            .meta("author", commit.author_name.as_str())
            .meta("email", commit.author_email.as_str())
            .meta("committed_at", commit.committed_at.as_str())
            .build();

        graph
            .add_episode(episode)
            .map_err(|e| format!("Failed to insert episode: {e}"))?;

        println!(
            "  {} {} \"{}\" — imported",
            "✓".green(),
            short_hash,
            truncate(&commit.subject, 60)
        );
        imported += 1;
    }

    println!();
    println!(
        "Imported {} new commit{}. {} already existed.",
        imported,
        if imported == 1 { "" } else { "s" },
        skipped
    );

    Ok(())
}

/// Install a post-commit hook in the given git repo.
pub fn install_post_commit_hook(repo: &Path) -> Result<(), String> {
    let hooks_dir = repo.join(".git/hooks");
    if !hooks_dir.exists() {
        return Err(
            "No .git/hooks directory found — is this a git repo?".to_string(),
        );
    }

    let hook_path = hooks_dir.join("post-commit");
    let content = "#!/bin/sh\nctxgraph watch --last 1\n";

    std::fs::write(&hook_path, content)
        .map_err(|e| format!("Failed to write hook: {e}"))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&hook_path)
            .map_err(|e| format!("Failed to read hook metadata: {e}"))?
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&hook_path, perms)
            .map_err(|e| format!("Failed to set hook permissions: {e}"))?;
    }

    println!("Installed post-commit hook: {}", hook_path.display());
    Ok(())
}

/// Top-level handler called from main.rs — opens the graph and runs watch.
pub fn run_watch_command(
    last: usize,
    since: Option<String>,
    repo: Option<PathBuf>,
    install_hook: bool,
) -> ctxgraph::Result<()> {
    let repo_path = repo
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap());

    if install_hook {
        install_post_commit_hook(&repo_path)
            .map_err(|e| ctxgraph::CtxGraphError::InvalidInput(e))?;
    }

    let graph = open_graph()?;
    run_watch(&graph, last, since.as_deref(), Some(&repo_path), false)
        .map_err(|e| ctxgraph::CtxGraphError::InvalidInput(e))
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars - 1).collect();
        format!("{truncated}…")
    }
}
