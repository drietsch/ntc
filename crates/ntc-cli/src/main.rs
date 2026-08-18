//! `ntc` — NTC-Web developer CLI.
//!
//! Subcommands:
//! - `gen-schemas`: regenerate the machine-readable contracts from the Rust
//!   source of truth (CI drift-checks the output).
//! - `schemac`: canonicalize raw tool schemas (JSONL in → JSONL out). The
//!   Python data pipeline shells out to this — the rendering has exactly one
//!   implementation.
//! - `verify`: parse + deep-verify a `.ntc` file, print the manifest.
//! - `fixture-gen`: write the tiny seeded random-weight `.ntc` fixture.
//! - `infer`: compile an utterance against tools with a `.ntc` model on the
//!   CPU reference backend (optionally dumping head logits).

use std::io::{BufRead, Write as _};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use schemars::schema_for;

#[derive(Parser)]
#[command(name = "ntc", version, about = "NTC-Web developer CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Regenerate contracts/*.schema.json from the Rust types.
    GenSchemas {
        /// Repository root (containing contracts/).
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Only print what would change; exit 1 on drift.
        #[arg(long)]
        check: bool,
    },
    /// Canonicalize raw tool schemas: JSONL on stdin (or --input), JSONL out.
    /// Each line: a RawToolSchema, or {"schema": {...}, "index": n}.
    Schemac {
        #[arg(long)]
        input: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Parse and deep-verify a .ntc file.
    Verify {
        file: PathBuf,
        /// Print the full manifest JSON (for conformance diffing).
        #[arg(long)]
        dump_manifest: bool,
    },
    /// Generate the tiny random-weight fixture model + example inputs.
    FixtureGen {
        #[arg(long, default_value = "fixtures/models/tiny-v1")]
        out: PathBuf,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Compile an utterance on the CPU reference backend.
    Infer {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        utterance: String,
        /// JSON file with an array of raw tool schemas.
        #[arg(long)]
        tools: PathBuf,
        #[arg(long)]
        timezone: Option<String>,
        /// RFC 3339 "now" override for deterministic output.
        #[arg(long)]
        now: Option<String>,
        /// Also print raw head logits.
        #[arg(long)]
        dump_heads: bool,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::GenSchemas { root, check } => gen_schemas(&root, check),
        Command::Schemac { input, output } => schemac(input, output),
        Command::Verify {
            file,
            dump_manifest,
        } => verify(&file, dump_manifest),
        Command::FixtureGen { out, seed } => fixture_gen(&out, seed),
        Command::Infer {
            model,
            utterance,
            tools,
            timezone,
            now,
            dump_heads,
        } => infer(&model, &utterance, &tools, timezone, now, dump_heads),
    }
}

fn gen_schemas(root: &std::path::Path, check: bool) -> Result<()> {
    let targets: Vec<(PathBuf, serde_json::Value)> = vec![
        (
            root.join("contracts/action-ir/v1/action-ir.schema.json"),
            serde_json::to_value(schema_for!(ntc_core::ir::ActionIr))?,
        ),
        (
            root.join("contracts/action-ir/v1/compile-request.schema.json"),
            serde_json::to_value(schema_for!(ntc_core::ir::CompileRequest))?,
        ),
        (
            root.join("contracts/action-ir/v1/compile-outcome.schema.json"),
            serde_json::to_value(schema_for!(ntc_runtime::CompileOutcome))?,
        ),
        (
            root.join("contracts/tool-abi/v1/tool-abi.schema.json"),
            serde_json::to_value(schema_for!(ntc_core::schema::CanonicalTool))?,
        ),
    ];
    let mut drift = false;
    for (path, value) in targets {
        let rendered = serde_json::to_string_pretty(&value)? + "\n";
        let existing = std::fs::read_to_string(&path).ok();
        if existing.as_deref() == Some(rendered.as_str()) {
            continue;
        }
        if check {
            eprintln!("drift: {}", path.display());
            drift = true;
        } else {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(&path, rendered)
                .with_context(|| format!("writing {}", path.display()))?;
            eprintln!("wrote {}", path.display());
        }
    }
    if drift {
        bail!("contract schemas have drifted; run `ntc gen-schemas`");
    }
    Ok(())
}

fn schemac(input: Option<PathBuf>, output: Option<PathBuf>) -> Result<()> {
    let reader: Box<dyn BufRead> = match input {
        Some(p) => Box::new(std::io::BufReader::new(std::fs::File::open(p)?)),
        None => Box::new(std::io::BufReader::new(std::io::stdin())),
    };
    let mut writer: Box<dyn std::io::Write> = match output {
        Some(p) => Box::new(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => Box::new(std::io::BufWriter::new(std::io::stdout())),
    };
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value =
            serde_json::from_str(&line).with_context(|| format!("line {}", lineno + 1))?;
        let (raw, index) = match value.get("schema") {
            Some(s) => (
                s.clone(),
                value.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize,
            ),
            None => (value, 0),
        };
        let raw: ntc_core::schema::RawToolSchema = serde_json::from_value(raw)
            .with_context(|| format!("line {}: not a raw tool schema", lineno + 1))?;
        let tool = ntc_core::schema::compile_schema(&raw)
            .map_err(|e| anyhow::anyhow!("line {}: {e}", lineno + 1))?;
        let text = tool.to_neural_text(index);
        let out = serde_json::json!({
            "id": tool.id,
            "abi_version": tool.abi_version,
            "index": index,
            "tool": tool,
            "text": text,
        });
        writeln!(writer, "{}", serde_json::to_string(&out)?)?;
    }
    Ok(())
}

fn verify(file: &std::path::Path, dump_manifest: bool) -> Result<()> {
    let bytes = std::fs::read(file).with_context(|| format!("reading {}", file.display()))?;
    let report = ntc_format::verify(&bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    if dump_manifest {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "OK {}: {} `{}` ({}), {} tensors, {:.1} MiB tensor data, tokenizer {} bytes",
            file.display(),
            report.architecture,
            report.model_version,
            report.quantization,
            report.tensor_count,
            report.total_tensor_bytes as f64 / (1024.0 * 1024.0),
            report.tokenizer_bytes,
        );
    }
    Ok(())
}

fn fixture_gen(out: &std::path::Path, seed: u64) -> Result<()> {
    use ntc_format::writer::NtcWriter;
    use ntc_format::{DType, NtcMetadata};
    use ntc_model::test_support::{random_weights, test_tokenizer_json, tiny_config};
    use ntc_model::weights::tensor_specs;

    let cfg = tiny_config();
    let weights = random_weights(&cfg, seed);

    let metadata = NtcMetadata {
        architecture: ntc_model::ARCHITECTURE.into(),
        model_version: format!("tiny-v1-seed{seed}"),
        ir_version: ntc_core::IR_VERSION,
        abi_version: ntc_core::ABI_VERSION,
        head_spec_version: 1,
        tokenizer_sha256: String::new(), // writer fills this
        quantization: "f32".into(),
        model: serde_json::to_value(&cfg)?,
        semantic_types: vec![],
    };
    let mut w = NtcWriter::new(metadata, test_tokenizer_json().into_bytes());
    for (name, shape) in tensor_specs(&cfg) {
        let t = weights.get(&name).map_err(|e| anyhow::anyhow!("{e}"))?;
        let bytes: Vec<u8> = t.data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        w.add_tensor(&name, DType::F32, &shape_u64, &bytes)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    let buf = w.finish();

    std::fs::create_dir_all(out)?;
    let model_path = out.join("tiny.ntc");
    std::fs::write(&model_path, &buf)?;

    // Self-check + manifest for conformance diffing.
    let report = ntc_format::verify(&buf).map_err(|e| anyhow::anyhow!("{e}"))?;
    std::fs::write(
        out.join("tiny.manifest.json"),
        serde_json::to_string_pretty(&report)? + "\n",
    )?;
    eprintln!(
        "wrote {} ({} tensors, {} bytes)",
        model_path.display(),
        report.tensor_count,
        buf.len()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn infer(
    model: &std::path::Path,
    utterance: &str,
    tools: &std::path::Path,
    timezone: Option<String>,
    now: Option<String>,
    dump_heads: bool,
) -> Result<()> {
    use ntc_runtime::{CompilerConfig, NeuralToolCompiler};

    let bytes = std::fs::read(model).with_context(|| format!("reading {}", model.display()))?;
    let mut config = CompilerConfig::default();
    if let Some(tz) = &timezone {
        config.timezone = tz.clone();
    }
    let mut compiler =
        NeuralToolCompiler::load_cpu(&bytes, config).map_err(|e| anyhow::anyhow!("{e}"))?;

    let tool_defs: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(tools)?)?;
    for def in tool_defs {
        let raw: ntc_core::schema::RawToolSchema = serde_json::from_value(def)?;
        compiler
            .register_tool(raw)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }

    let req = ntc_core::ir::CompileRequest {
        utterance: utterance.to_string(),
        locale: None,
        timezone,
        now,
        candidates: None,
        context: None,
    };
    if dump_heads {
        let heads = compiler
            .run_heads(&req)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("{}", serde_json::to_string(&heads)?);
        return Ok(());
    }
    let outcome = compiler.compile(&req).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{}", serde_json::to_string_pretty(&outcome)?);
    Ok(())
}
