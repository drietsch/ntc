//! WGSL validation (no GPU needed), kernel-level parity vs `ntc_model::ops`,
//! and full-backend parity vs `CpuRefBackend` on the tiny fixture model.
//!
//! GPU tests skip gracefully (eprintln + return) when no adapter is
//! available so CI without a GPU still passes.

use ntc_core::schema::{compile_schema, RawToolSchema};
use ntc_core::tokenizer::NtcTokenizer;
use ntc_model::test_support::{random_weights, test_tokenizer_json, tiny_config};
use ntc_model::{ops, Backend, CpuRefBackend, ModelInputs, Tensor};
use ntc_webgpu::{GpuExecutor, WgpuBackend, WgpuContext};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Cross-backend tolerance: abs diff <= 1e-4 + 1e-3 * |ref|.
fn assert_close(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 1e-4 + 1e-3 * w.abs();
        assert!(
            (g - w).abs() <= tol,
            "{name}[{i}]: got {g}, want {w} (tol {tol})"
        );
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("no NaN logits"))
        .expect("non-empty")
        .0
}

fn rand_tensor(rng: &mut ChaCha8Rng, shape: &[usize], scale: f32) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec(
        shape,
        (0..n).map(|_| rng.gen_range(-scale..scale)).collect(),
    )
}

/// GPU executor, or None (with an eprintln) when no adapter exists.
fn executor() -> Option<GpuExecutor> {
    match pollster::block_on(WgpuContext::new()) {
        Ok(ctx) => {
            eprintln!(
                "GPU test on adapter `{}` ({:?})",
                ctx.caps.adapter_name, ctx.caps.backend
            );
            Some(GpuExecutor::new(ctx))
        }
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Shader validation (runs everywhere, no GPU required)
// ---------------------------------------------------------------------------

#[test]
fn wgsl_kernels_validate_with_naga() {
    for (name, src) in ntc_webgpu::kernels::ALL {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{name}: WGSL parse error:\n{}", e.emit_to_string(src)));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: WGSL validation error: {e:?}"));
    }
}

// ---------------------------------------------------------------------------
// Kernel-level parity
// ---------------------------------------------------------------------------

#[test]
fn matmul_bias_parity() {
    let Some(mut ex) = executor() else { return };
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let x = rand_tensor(&mut rng, &[7, 13], 1.0);
    let w = rand_tensor(&mut rng, &[13, 9], 1.0);
    let b = rand_tensor(&mut rng, &[9], 1.0);

    let want = ops::linear(&x, &w, Some(&b));
    let got = ex.matmul_host(&x, &w, Some(&b)).unwrap();
    assert_eq!(got.shape, want.shape);
    assert_close("matmul+bias", &got.data, &want.data);

    let want = ops::linear(&x, &w, None);
    let got = ex.matmul_host(&x, &w, None).unwrap();
    assert_close("matmul(no bias)", &got.data, &want.data);
}

#[test]
fn layernorm_parity() {
    let Some(mut ex) = executor() else { return };
    let mut rng = ChaCha8Rng::seed_from_u64(12);
    let x = rand_tensor(&mut rng, &[5, 32], 2.0);
    let gamma = rand_tensor(&mut rng, &[32], 1.0);
    let beta = rand_tensor(&mut rng, &[32], 1.0);

    let want = ops::layer_norm(&x, &gamma, &beta, 1e-5);
    let got = ex.layer_norm_host(&x, &gamma, &beta, 1e-5).unwrap();
    assert_eq!(got.shape, want.shape);
    assert_close("layernorm", &got.data, &want.data);
}

#[test]
fn gelu_parity() {
    let Some(mut ex) = executor() else { return };
    let mut rng = ChaCha8Rng::seed_from_u64(13);
    let x = rand_tensor(&mut rng, &[4, 33], 3.0);

    let mut want = x.clone();
    ops::gelu(&mut want);
    let got = ex.gelu_host(&x).unwrap();
    assert_close("gelu", &got.data, &want.data);
}

#[test]
fn residual_add_parity() {
    let Some(mut ex) = executor() else { return };
    let mut rng = ChaCha8Rng::seed_from_u64(14);
    let x = rand_tensor(&mut rng, &[6, 21], 2.0);
    let y = rand_tensor(&mut rng, &[6, 21], 2.0);

    let mut want = x.clone();
    ops::add_inplace(&mut want, &y);
    let got = ex.add_host(&x, &y).unwrap();
    assert_close("residual add", &got.data, &want.data);
}

#[test]
fn softmax_parity() {
    let Some(mut ex) = executor() else { return };
    let mut rng = ChaCha8Rng::seed_from_u64(15);
    let mut x = rand_tensor(&mut rng, &[6, 17], 4.0);
    // Sprinkle masked (f32::MIN) entries like masked attention scores.
    for i in 0..6 {
        x.data[i * 17 + (i * 3) % 17] = f32::MIN;
        x.data[i * 17 + 16] = f32::MIN;
    }

    let mut want = x.clone();
    ops::softmax_rows(&mut want);
    let got = ex.softmax_host(&x).unwrap();
    assert_close("softmax", &got.data, &want.data);
    // Masked entries must be exactly zero on both sides.
    for i in 0..6 {
        assert_eq!(got.data[i * 17 + 16], 0.0);
    }
}

#[test]
fn attention_parity() {
    let Some(mut ex) = executor() else { return };
    let mut rng = ChaCha8Rng::seed_from_u64(16);
    let (lq, lkv, h, heads) = (6, 9, 32, 4);
    let q_states = rand_tensor(&mut rng, &[lq, h], 1.0);
    let kv_states = rand_tensor(&mut rng, &[lkv, h], 1.0);
    let mut mask = vec![true; lkv];
    mask[4] = false;
    mask[8] = false;
    let scale = 0.5 / (h as f32).sqrt();
    let mk = |rng: &mut ChaCha8Rng| {
        (
            rand_tensor(rng, &[h, h], scale),
            rand_tensor(rng, &[h], scale),
        )
    };
    let (wq, wk, wv, wo) = (mk(&mut rng), mk(&mut rng), mk(&mut rng), mk(&mut rng));

    let want = ops::attention(
        &q_states,
        &kv_states,
        &mask,
        (&wq.0, &wq.1),
        (&wk.0, &wk.1),
        (&wv.0, &wv.1),
        (&wo.0, &wo.1),
        heads,
    );
    let got = ex
        .attention_host(
            &q_states,
            &kv_states,
            &mask,
            (&wq.0, &wq.1),
            (&wk.0, &wk.1),
            (&wv.0, &wv.1),
            (&wo.0, &wo.1),
            heads,
        )
        .unwrap();
    assert_eq!(got.shape, want.shape);
    assert_close("attention", &got.data, &want.data);
}

// ---------------------------------------------------------------------------
// Full-backend parity (same fixture as crates/ntc-model/tests/cpu_forward.rs)
// ---------------------------------------------------------------------------

fn tools() -> Vec<ntc_core::CanonicalTool> {
    let calendar: RawToolSchema = serde_json::from_value(serde_json::json!({
        "name": "calendar.create",
        "description": "create a calendar event",
        "parameters": {
            "title": {"type": "string", "required": true},
            "start": {"type": "string", "format": "date-time", "required": true},
            "priority": {"type": "string", "enum": ["low", "normal", "high"]}
        }
    }))
    .unwrap();
    let email: RawToolSchema = serde_json::from_value(serde_json::json!({
        "name": "email.send",
        "description": "send an email",
        "parameters": {
            "recipient": {"type": "string", "required": true},
            "subject": {"type": "string"}
        }
    }))
    .unwrap();
    vec![
        compile_schema(&calendar).unwrap(),
        compile_schema(&email).unwrap(),
    ]
}

#[test]
fn full_backend_parity() {
    let ctx = match pollster::block_on(WgpuContext::new()) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            return;
        }
    };

    let cfg = tiny_config();
    let tokenizer = NtcTokenizer::from_bytes(test_tokenizer_json().as_bytes()).unwrap();
    let tools = tools();
    let refs: Vec<&_> = tools.iter().collect();
    let utterance = tokenizer
        .encode_utterance("make a dentist appointment tomorrow afternoon")
        .unwrap();
    let inputs = ModelInputs::pack(&cfg, &tokenizer, &utterance, &refs).unwrap();

    let weights = random_weights(&cfg, 7);
    let mut gpu = WgpuBackend::new(cfg.clone(), &weights, ctx).unwrap();
    let mut cpu = CpuRefBackend::new(cfg.clone(), weights);

    let cpu_out = cpu.run(&inputs).unwrap();
    let gpu_out = gpu.run(&inputs).unwrap();

    // (a) Same tensor set, same shapes.
    let mut cpu_names: Vec<&String> = cpu_out.tensors.keys().collect();
    let mut gpu_names: Vec<&String> = gpu_out.tensors.keys().collect();
    cpu_names.sort();
    gpu_names.sort();
    assert_eq!(cpu_names, gpu_names, "head output tensor sets differ");
    for name in &cpu_names {
        assert_eq!(
            gpu_out.tensors[*name].shape, cpu_out.tensors[*name].shape,
            "`{name}` shape mismatch"
        );
    }

    // (b) Element-wise tolerance everywhere (fill values are the identical
    // f32::MIN / 0.0 constants on both backends, so the whole tensor is
    // comparable; the valid region is what actually exercises the GPU).
    for name in &cpu_names {
        assert_close(
            name,
            &gpu_out.tensors[*name].data,
            &cpu_out.tensors[*name].data,
        );
    }

    // (c) Decision parity.
    let cpu_t = |n: &str| cpu_out.get(n).unwrap();
    let gpu_t = |n: &str| gpu_out.get(n).unwrap();
    assert_eq!(
        argmax(&gpu_t("action.logits").data),
        argmax(&cpu_t("action.logits").data),
        "action argmax differs"
    );
    assert_eq!(
        argmax(&gpu_t("tool.logits").data),
        argmax(&cpu_t("tool.logits").data),
        "tool argmax differs"
    );

    let a = cfg.max_args;
    for (t, tool) in inputs.tools.iter().enumerate() {
        for k in 0..tool.arg_anchors.len() {
            let base = t * a + k;
            let slice = |t4: &Tensor, c: usize| t4.data[base * c..(base + 1) * c].to_vec();

            assert_eq!(
                argmax(&slice(gpu_t("presence.logits"), 4)),
                argmax(&slice(cpu_t("presence.logits"), 4)),
                "presence argmax differs (tool {t}, arg {k})"
            );
            assert_eq!(
                argmax(&slice(gpu_t("datetime.relation.logits"), 10)),
                argmax(&slice(cpu_t("datetime.relation.logits"), 10)),
                "datetime.relation argmax differs (tool {t}, arg {k})"
            );

            let n_enum = tool.enum_anchors[k].len();
            if n_enum > 0 {
                let e = cfg.max_enum_values;
                let g = &gpu_t("enum.logits").data[base * e..base * e + n_enum];
                let c = &cpu_t("enum.logits").data[base * e..base * e + n_enum];
                assert_eq!(
                    argmax(g),
                    argmax(c),
                    "enum argmax differs (tool {t}, arg {k})"
                );
            }
        }
    }
}
