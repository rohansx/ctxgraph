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

    /// Auto-capture git commits as episodes
    Watch {
        /// Import the last N commits
        #[arg(long, default_value = "10")]
        last: usize,

        /// Only import commits since this date (ISO 8601)
        #[arg(long)]
        since: Option<String>,

        /// Path to git repository (default: current directory)
        #[arg(long)]
        repo: Option<std::path::PathBuf>,

        /// Install as .git/hooks/post-commit hook
        #[arg(long)]
        install_hook: bool,
    },
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
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Init { name } => commands::init::run(name),
        Commands::Log { text, source, tags } => commands::log::run(text, source, tags),
        Commands::Query {
            text,
            limit,
            after,
            source,
        } => commands::query::run(text, limit, after, source),
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
        Commands::Watch {
            last,
            since,
            repo,
            install_hook,
        } => commands::watch::run_watch_command(last, since, repo, install_hook),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
