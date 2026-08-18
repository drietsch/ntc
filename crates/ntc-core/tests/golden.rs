//! Golden-fixture conformance: canonical ABI rendering byte-stability and
//! IR accept/reject parity. The same fixtures are consumed by the Python
//! side's CI — a diff here is a contract change and requires an ABI/IR
//! version bump (contracts/VERSIONS.md).

use std::path::PathBuf;

use ntc_core::schema::{compile_schema, RawToolSchema};

fn fixtures(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(sub)
}

#[test]
fn schema_abi_rendering_is_byte_stable() {
    let dir = fixtures("schema-abi");
    let mut checked = 0;
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("fixtures/schema-abi exists")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    entries.sort();
    assert!(!entries.is_empty(), "no schema-abi fixtures found");
    for path in entries {
        let raw: RawToolSchema = serde_json::from_str(&std::fs::read_to_string(&path).unwrap())
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let tool = compile_schema(&raw).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let rendered = tool.to_neural_text(0);
        let golden_path = path.with_extension("canon.txt");
        let golden = std::fs::read_to_string(&golden_path)
            .unwrap_or_else(|e| panic!("{}: {e}", golden_path.display()));
        assert_eq!(
            rendered,
            golden.trim_end_matches('\n'),
            "canonical rendering drifted for {} — this is a contract change",
            path.display()
        );
        checked += 1;
    }
    assert!(
        checked >= 10,
        "expected ≥10 golden schemas, found {checked}"
    );
}

#[test]
fn ir_fixtures_accept_reject_parity() {
    let valid_dir = fixtures("ir/valid");
    let invalid_dir = fixtures("ir/invalid");
    let mut n = 0;
    for path in std::fs::read_dir(&valid_dir)
        .unwrap()
        .map(|e| e.unwrap().path())
    {
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        serde_json::from_str::<ntc_core::ir::ActionIr>(&text)
            .unwrap_or_else(|e| panic!("{} must parse: {e}", path.display()));
        n += 1;
    }
    for path in std::fs::read_dir(&invalid_dir)
        .unwrap()
        .map(|e| e.unwrap().path())
    {
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            serde_json::from_str::<ntc_core::ir::ActionIr>(&text).is_err(),
            "{} must be rejected",
            path.display()
        );
        n += 1;
    }
    assert!(n >= 6, "expected ≥6 IR fixtures, found {n}");
}
