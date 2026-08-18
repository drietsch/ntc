//! Tokenizer parity: the frozen artifact (contracts/tokenizer/tokenizer.json)
//! must produce exactly the golden ids + byte offsets recorded by the Python
//! side (fixtures/tokenizer/vectors.jsonl). Same underlying Rust core, but
//! this pins version skew and offset-space conventions.

use std::path::PathBuf;

use ntc_core::tokenizer::NtcTokenizer;

fn repo(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path)
}

#[derive(serde::Deserialize)]
struct Vector {
    text: String,
    ids: Vec<u32>,
    offsets: Vec<(usize, usize)>,
}

#[test]
fn frozen_tokenizer_matches_golden_vectors() {
    let tok_path = repo("contracts/tokenizer/tokenizer.json");
    let vec_path = repo("fixtures/tokenizer/vectors.jsonl");
    let (Ok(tok_bytes), Ok(vectors)) =
        (std::fs::read(&tok_path), std::fs::read_to_string(&vec_path))
    else {
        panic!(
            "tokenizer contract not frozen: expected {} and {}",
            tok_path.display(),
            vec_path.display()
        );
    };
    let tokenizer = NtcTokenizer::from_bytes(&tok_bytes).expect("frozen tokenizer loads");

    let mut checked = 0;
    for line in vectors.lines().filter(|l| !l.trim().is_empty()) {
        let v: Vector = serde_json::from_str(line).unwrap();
        let seq = tokenizer.encode_utterance(&v.text).unwrap();
        assert_eq!(seq.ids, v.ids, "id mismatch for {:?}", v.text);
        assert_eq!(seq.offsets, v.offsets, "offset mismatch for {:?}", v.text);
        checked += 1;
    }
    assert!(
        checked >= 15,
        "expected ≥15 golden vectors, found {checked}"
    );
}
