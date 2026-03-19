use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

use sha2::{Digest, Sha256};

/// Specification for a downloadable ONNX model.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: String,
    pub url: String,
    pub sha256: String,
    pub size_bytes: u64,
}

/// Manages downloading, caching, and verifying ONNX model files.
pub struct ModelManager {
    cache_dir: PathBuf,
}

impl ModelManager {
    /// Create a new `ModelManager` using the default cache directory
    /// (`~/.cache/ctxgraph/models/`).
    pub fn new() -> Result<Self, ModelManagerError> {
        let cache = Self::default_cache_dir()?;
        Ok(Self { cache_dir: cache })
    }

    /// Create a `ModelManager` with a custom cache directory (useful for tests).
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self, ModelManagerError> {
        fs::create_dir_all(&cache_dir).map_err(|e| ModelManagerError::Io {
            context: format!("creating cache dir {}", cache_dir.display()),
            source: e,
        })?;
        Ok(Self { cache_dir })
    }

    /// Return the default cache directory (`~/.cache/ctxgraph/models/`),
    /// creating it if it does not exist.
    pub fn default_cache_dir() -> Result<PathBuf, ModelManagerError> {
        let base = dirs::cache_dir().ok_or(ModelManagerError::NoCacheDir)?;
        let dir = base.join("ctxgraph").join("models");
        fs::create_dir_all(&dir).map_err(|e| ModelManagerError::Io {
            context: format!("creating cache dir {}", dir.display()),
            source: e,
        })?;
        Ok(dir)
    }

    /// Path where a given model would be stored locally.
    pub fn model_path(&self, spec: &ModelSpec) -> PathBuf {
        self.cache_dir.join(&spec.name)
    }

    /// Check whether the model file exists on disk and its size matches the spec.
    pub fn is_cached(&self, spec: &ModelSpec) -> bool {
        let path = self.model_path(spec);
        match fs::metadata(&path) {
            Ok(meta) => meta.len() == spec.size_bytes,
            Err(_) => false,
        }
    }

    /// Verify the SHA-256 hash of a cached model file.
    /// Returns `Ok(true)` if the hash matches, `Ok(false)` if it doesn't,
    /// or an error if the file cannot be read.
    pub fn verify(&self, spec: &ModelSpec) -> Result<bool, ModelManagerError> {
        let path = self.model_path(spec);
        let mut file = fs::File::open(&path).map_err(|e| ModelManagerError::Io {
            context: format!("opening {} for verification", path.display()),
            source: e,
        })?;

        let mut hasher = Sha256::new();
        let mut buf = [0u8; 8192];
        loop {
            let n = file.read(&mut buf).map_err(|e| ModelManagerError::Io {
                context: "reading file for hash".into(),
                source: e,
            })?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }

        let digest = format!("{:x}", hasher.finalize());
        Ok(digest == spec.sha256)
    }

    /// Download a model, verify its hash, and return the local path.
    pub fn download(&self, spec: &ModelSpec) -> Result<PathBuf, ModelManagerError> {
        let dest = self.model_path(spec);

        let response = reqwest::blocking::get(&spec.url).map_err(|e| {
            ModelManagerError::Download {
                url: spec.url.clone(),
                source: e,
            }
        })?;

        if !response.status().is_success() {
            return Err(ModelManagerError::HttpStatus {
                url: spec.url.clone(),
                status: response.status().as_u16(),
            });
        }

        let total_size = response.content_length().unwrap_or(spec.size_bytes);

        let pb = indicatif::ProgressBar::new(total_size);
        pb.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut file = fs::File::create(&dest).map_err(|e| ModelManagerError::Io {
            context: format!("creating {}", dest.display()),
            source: e,
        })?;

        let mut downloaded: u64 = 0;
        let mut reader = response;
        let mut buf = [0u8; 8192];
        loop {
            let n = reader.read(&mut buf).map_err(|e| ModelManagerError::Io {
                context: "reading download stream".into(),
                source: e,
            })?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).map_err(|e| ModelManagerError::Io {
                context: "writing model file".into(),
                source: e,
            })?;
            downloaded += n as u64;
            pb.set_position(downloaded);
        }
        pb.finish_with_message("download complete");

        // Verify hash after download
        let ok = self.verify(spec)?;
        if !ok {
            // Remove the corrupt file
            let _ = fs::remove_file(&dest);
            return Err(ModelManagerError::HashMismatch {
                model: spec.name.clone(),
            });
        }

        Ok(dest)
    }

    /// Return the cached model path if it exists and is valid, otherwise download it.
    pub fn get_or_download(&self, spec: &ModelSpec) -> Result<PathBuf, ModelManagerError> {
        if self.is_cached(spec) {
            // Optionally verify hash of cached file
            if self.verify(spec)? {
                return Ok(self.model_path(spec));
            }
        }
        self.download(spec)
    }
}

// ---------------------------------------------------------------------------
// Pre-defined model specs
// ---------------------------------------------------------------------------

/// GLiNER Large v2.1 INT8 quantized model (span-based NER).
///
/// From: <https://huggingface.co/onnx-community/gliner_large-v2.1>
pub fn gliner_large_v21_int8() -> ModelSpec {
    ModelSpec {
        name: "gliner_large-v2.1/onnx/model_int8.onnx".into(),
        url: "https://huggingface.co/onnx-community/gliner_large-v2.1/resolve/main/onnx/model_int8.onnx".into(),
        sha256: "pending_verification".into(),
        size_bytes: 653_000_000,
    }
}

/// GLiNER Large v2.1 tokenizer.
pub fn gliner_large_v21_tokenizer() -> ModelSpec {
    ModelSpec {
        name: "gliner_large-v2.1/tokenizer.json".into(),
        url: "https://huggingface.co/onnx-community/gliner_large-v2.1/resolve/main/tokenizer.json".into(),
        sha256: "pending_verification".into(),
        size_bytes: 17_000_000,
    }
}

/// GLiNER Multitask Large v0.5 (token-based NER + relation extraction).
///
/// NOTE: This model requires ONNX conversion from PyTorch. No pre-built ONNX
/// export exists on HuggingFace. Use `scripts/convert_model.py` to convert.
/// From: <https://huggingface.co/knowledgator/gliner-multitask-large-v0.5>
pub fn gliner_multitask_large() -> ModelSpec {
    ModelSpec {
        name: "gliner-multitask-large-v0.5/onnx/model.onnx".into(),
        url: "https://huggingface.co/knowledgator/gliner-multitask-large-v0.5/resolve/main/onnx/model.onnx".into(),
        sha256: "pending_conversion".into(),
        size_bytes: 1_760_000_000,
    }
}

/// GLiNER Multitask Large v0.5 tokenizer.
pub fn gliner_multitask_tokenizer() -> ModelSpec {
    ModelSpec {
        name: "gliner-multitask-large-v0.5/tokenizer.json".into(),
        url: "https://huggingface.co/knowledgator/gliner-multitask-large-v0.5/resolve/main/tokenizer.json".into(),
        sha256: "pending_verification".into(),
        size_bytes: 8_660_000,
    }
}

/// MiniLM L6 v2 sentence-embedding model (for v0.3 semantic search).
pub fn minilm_l6_v2() -> ModelSpec {
    ModelSpec {
        name: "minilm-l6-v2.onnx".into(),
        url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx".into(),
        sha256: "pending_verification".into(),
        size_bytes: 80_000_000,
    }
}

// ---------------------------------------------------------------------------
// Convenience: download all models needed for NER extraction
// ---------------------------------------------------------------------------

impl ModelManager {
    /// Download the NER model and tokenizer needed for Tier 1 extraction.
    ///
    /// Downloads `gliner_large-v2.1` INT8 model + tokenizer to the cache directory.
    pub fn ensure_ner_models(&self) -> Result<(PathBuf, PathBuf), ModelManagerError> {
        let model = self.get_or_download(&gliner_large_v21_int8())?;
        let tokenizer = self.get_or_download(&gliner_large_v21_tokenizer())?;
        Ok((model, tokenizer))
    }

    /// Download the multitask model and tokenizer needed for relation extraction.
    ///
    /// Returns `None` if the model is not available (needs ONNX conversion).
    pub fn ensure_rel_models(&self) -> Option<(PathBuf, PathBuf)> {
        let model = self.get_or_download(&gliner_multitask_large()).ok()?;
        let tokenizer = self.get_or_download(&gliner_multitask_tokenizer()).ok()?;
        Some((model, tokenizer))
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ModelManagerError {
    #[error("could not determine cache directory")]
    NoCacheDir,

    #[error("I/O error ({context}): {source}")]
    Io {
        context: String,
        source: std::io::Error,
    },

    #[error("download failed for {url}: {source}")]
    Download {
        url: String,
        source: reqwest::Error,
    },

    #[error("HTTP {status} for {url}")]
    HttpStatus { url: String, status: u16 },

    #[error("SHA-256 hash mismatch for {model}")]
    HashMismatch { model: String },
}
