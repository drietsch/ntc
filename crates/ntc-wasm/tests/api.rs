//! Native smoke test for the wasm wrapper. `NtcWeb` compiles on native
//! because `#[wasm_bindgen]` is inert off-wasm; only the error path touches
//! `JsValue`, and this test stays on the happy path.
#![cfg(not(target_arch = "wasm32"))]

use ntc_wasm::NtcWeb;

fn model_bytes() -> Vec<u8> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/models/tiny-v1/tiny.ntc"
    );
    std::fs::read(path).expect("read fixtures/models/tiny-v1/tiny.ntc")
}

#[test]
fn version_matches_workspace() {
    assert_eq!(ntc_wasm::version(), env!("CARGO_PKG_VERSION"));
}

#[test]
fn load_register_compile_smoke() {
    let bytes = model_bytes();
    let config = r#"{"timezone": "Europe/Berlin", "locale": "de-DE"}"#;
    let mut ntc =
        NtcWeb::new(&bytes, Some(config.to_string())).expect("model loads on CPU backend");

    let calendar = serde_json::json!({
        "name": "calendar.create",
        "description": "create a calendar event",
        "parameters": {
            "title": {"type": "string", "required": true},
            "start": {"type": "string", "format": "date-time", "required": true},
            "duration_minutes": {"type": "integer", "semantic": "DURATION"}
        }
    });
    let email = serde_json::json!({
        "name": "email.send",
        "description": "send an email",
        "parameters": {
            "recipient": {"type": "string", "required": true},
            "subject": {"type": "string"}
        }
    });
    let id0 = ntc
        .register_tool(calendar.to_string())
        .expect("register calendar.create");
    let id1 = ntc
        .register_tool(email.to_string())
        .expect("register email.send");
    assert_eq!((id0, id1), (0, 1), "tool ids are registry indices");

    let request = serde_json::json!({
        "utterance": "make a dentist appointment tomorrow afternoon",
        "timezone": "Europe/Berlin",
        "now": "2026-08-18T11:00:00+02:00"
    });
    let out = ntc.compile(request.to_string()).expect("compile succeeds");

    let value: serde_json::Value =
        serde_json::from_str(&out).expect("compile output is valid JSON");
    let outcome = value
        .get("outcome")
        .and_then(|o| o.as_str())
        .expect("output has an `outcome` field");
    assert!(
        matches!(outcome, "CALL" | "ASK" | "NO_CALL"),
        "unexpected outcome tag: {outcome}"
    );

    // Determinism: same request, same JSON.
    let again = ntc
        .compile(request.to_string())
        .expect("recompile succeeds");
    assert_eq!(out, again, "compilation must be deterministic");
}
