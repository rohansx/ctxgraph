mod commands;
mod display;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ctxgraph", about = "Local-first context graph engine")]
#[command(version, propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize ctxgraph in the current directory
    Init {
        /// Project name
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Log a decision or event
    Log {
        /// The text to log
        text: String,

        /// Source of this information
        #[arg(short, long)]
        source: Option<String>,

        /// Comma-separated tags
        #[arg(short, long)]
        tags: Option<String>,

        /// Use the v0.3 universal schema + LLM-first pipeline
        /// (loads schemas/universal.toml + prompts/extract.txt instead of
        /// the legacy GLiNER + tech-schema path). Requires an LLM env var:
        /// OPENROUTER_API_KEY / CEREBRAS_API_KEY / OPENAI_API_KEY /
        /// ANTHROPIC_API_KEY, or a running Ollama instance.
        #[arg(long)]
        universal: bool,
    },

    /// Search the context graph
    Query {
        /// Search query text
        text: String,

        /// Maximum results to return
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Only show results after this date (ISO-8601)
        #[arg(long)]
        after: Option<String>,

        /// Filter by source
        #[arg(long)]
        source: Option<String>,

        /// Use the v0.3 universal read path (verb → typed relation via
        /// embedding match, then deterministic SQL traversal). Falls back to
        /// fused search if the verb/entity can't be resolved.
        #[arg(long)]
        universal: bool,
    },

    /// List and show entities
    Entities {
        #[command(subcommand)]
        action: EntitiesAction,
    },

    /// List and show decisions
    Decisions {
        #[command(subcommand)]
        action: DecisionsAction,
    },

    /// Show graph statistics
    Stats,

    /// Manage ONNX models
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },

    /// Run the MCP server (JSON-RPC over stdio)
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },
}

#[derive(Subcommand)]
enum McpAction {
    /// Start the MCP server on stdio
    Start {
        /// Path to the graph database (overrides CTXGRAPH_DB env var)
        #[arg(long)]
        db: Option<String>,
    },
}

#[derive(Subcommand)]
enum ModelsAction {
    /// Download ONNX models required for extraction
    Download,
}

#[derive(Subcommand)]
enum EntitiesAction {
    /// List all entities
    List {
        /// Filter by entity type
        #[arg(short = 't', long = "type")]
        entity_type: Option<String>,

        /// Maximum results
        #[arg(short, long, default_value = "50")]
        limit: usize,
    },

    /// Show details for a specific entity
    Show {
        /// Entity ID or name
        id: String,
    },
}

#[derive(Subcommand)]
enum DecisionsAction {
    /// List all decisions
    List {
        /// Only show decisions after this date
        #[arg(long)]
        after: Option<String>,

        /// Filter by source
        #[arg(long)]
        source: Option<String>,

        /// Maximum results
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },

    /// Show full decision trace
    Show {
        /// Decision/episode ID
        id: String,
    },
}

fn main() {
    dotenvy::dotenv().ok();
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Init { name } => commands::init::run(name),
        Commands::Log {
            text,
            source,
            tags,
            universal,
        } => {
            if universal {
                commands::log_universal::run(text, source, tags)
            } else {
                commands::log::run(text, source, tags)
            }
        }
        Commands::Query {
            text,
            limit,
            after,
            source,
            universal,
        } => {
            if universal {
                commands::query_universal::run(text, limit)
            } else {
                commands::query::run(text, limit, after, source)
            }
        }
        Commands::Entities { action } => match action {
            EntitiesAction::List { entity_type, limit } => {
                commands::entities::list(entity_type, limit)
            }
            EntitiesAction::Show { id } => commands::entities::show(id),
        },
        Commands::Decisions { action } => match action {
            DecisionsAction::List {
                after,
                source,
                limit,
            } => commands::decisions::list(after, source, limit),
            DecisionsAction::Show { id } => commands::decisions::show(id),
        },
        Commands::Stats => commands::stats::run(),
        Commands::Models { action } => match action {
            ModelsAction::Download => commands::models::download(),
        },
        Commands::Mcp { action } => match action {
            McpAction::Start { db } => commands::mcp::start(db),
        },
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
