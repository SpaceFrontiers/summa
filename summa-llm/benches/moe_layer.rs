//! Reproducible single-GPU MoE layer benchmark.
//!
//! Run on CUDA with:
//! `cargo bench -p summa-llm --bench moe_layer --features training-fusion -- --tokens 8192`

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
fn main() {
    eprintln!("moe_layer requires Linux and --features training-fusion");
    std::process::exit(2);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn main() -> anyhow::Result<()> {
    benchmark::run()
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
mod benchmark {
    use std::path::PathBuf;
    use std::time::Instant;

    use anyhow::{Context, Result, bail};
    use burn::module::AutodiffModule;
    use burn::prelude::*;
    use burn::tensor::{DType, Distribution};
    use serde::Serialize;
    use summa_llm::model::FeedForward;
    use summa_llm::{default_device, parse_mal_file};

    #[derive(Debug)]
    struct Args {
        model: PathBuf,
        tokens: usize,
        warmup: usize,
        iterations: usize,
    }

    impl Default for Args {
        fn default() -> Self {
            Self {
                model: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../summa-mal/well-known/retriever_200m_moe.mal"),
                tokens: 8_192,
                warmup: 5,
                iterations: 20,
            }
        }
    }

    #[derive(Serialize)]
    struct Measurement {
        mode: &'static str,
        median_ms: f64,
        min_ms: f64,
        tokens_per_second: f64,
    }

    #[derive(Serialize)]
    struct Report {
        implementation: &'static str,
        model: String,
        device: String,
        dtype: &'static str,
        hidden_size: usize,
        intermediate_size: usize,
        experts: usize,
        top_k: usize,
        shared_experts: usize,
        tokens: usize,
        warmup: usize,
        iterations: usize,
        measurements: Vec<Measurement>,
    }

    pub fn run() -> Result<()> {
        let args = parse_args()?;
        let config = parse_mal_file(&args.model)
            .with_context(|| format!("failed to parse {}", args.model.display()))?;
        let block = (0..config.num_layers)
            .map(|layer| config.block_for_layer(layer))
            .find(|block| block.ffn.moe.is_some())
            .context("model has no MoE layer")?;
        let moe = block.ffn.moe.as_ref().expect("checked above");
        let intermediate = block.intermediate_size(config.hidden_size);

        let base_device = default_device();
        let training_device = base_device.clone().autodiff();
        training_device.seed(17);
        let layer = FeedForward::new(&config, block, &training_device);
        let training_input = Tensor::<2>::random(
            [args.tokens, config.hidden_size],
            Distribution::Default,
            &training_device,
        )
        .cast(DType::BF16);

        let inference_layer = layer.clone().valid();
        base_device.seed(17);
        let inference_input = Tensor::<2>::random(
            [args.tokens, config.hidden_size],
            Distribution::Default,
            &base_device,
        )
        .cast(DType::BF16);
        let forward = measure(
            args.warmup,
            args.iterations,
            args.tokens,
            &base_device,
            || {
                let output = inference_layer.forward(inference_input.clone());
                // Keep the result live until the synchronized iteration ends.
                std::hint::black_box(output);
            },
            "forward",
        )?;

        let training_core = measure(
            args.warmup,
            args.iterations,
            args.tokens,
            &training_device,
            || {
                let tracked = training_input.clone().detach().require_grad();
                let output = layer.forward(tracked.clone());
                let loss = output.cast(DType::F32).square().mean();
                let mut gradients = loss.backward();
                let input_gradient = tracked
                    .grad_remove(&mut gradients)
                    .expect("tracked input must receive a gradient");
                std::hint::black_box((gradients, input_gradient));
            },
            "forward_backward_core",
        )?;

        let training_with_router_losses = measure(
            args.warmup,
            args.iterations,
            args.tokens,
            &training_device,
            || {
                let tracked = training_input.clone().detach().require_grad();
                let (output, auxiliary) = layer.forward_with_aux(tracked.clone());
                let loss = output.cast(DType::F32).square().mean()
                    + auxiliary.expect("benchmark MoE config has router losses");
                let mut gradients = loss.backward();
                let input_gradient = tracked
                    .grad_remove(&mut gradients)
                    .expect("tracked input must receive a gradient");
                std::hint::black_box((gradients, input_gradient));
            },
            "forward_backward_with_router_losses",
        )?;

        let report = Report {
            implementation: "burn_grouped_fused",
            model: config.name.clone(),
            device: format!("{base_device:?}"),
            dtype: "bfloat16_tensor_cores_fp32_router",
            hidden_size: config.hidden_size,
            intermediate_size: intermediate,
            experts: moe.experts,
            top_k: moe.top_k,
            shared_experts: moe.shared_experts,
            tokens: args.tokens,
            warmup: args.warmup,
            iterations: args.iterations,
            measurements: vec![forward, training_core, training_with_router_losses],
        };
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    }

    fn measure(
        warmup: usize,
        iterations: usize,
        tokens: usize,
        device: &Device,
        mut operation: impl FnMut(),
        mode: &'static str,
    ) -> Result<Measurement> {
        for _ in 0..warmup {
            operation();
            device.sync()?;
        }
        let mut elapsed = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            device.sync()?;
            let started = Instant::now();
            operation();
            device.sync()?;
            elapsed.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        elapsed.sort_by(f64::total_cmp);
        let median_ms = if iterations % 2 == 0 {
            (elapsed[iterations / 2 - 1] + elapsed[iterations / 2]) / 2.0
        } else {
            elapsed[iterations / 2]
        };
        Ok(Measurement {
            mode,
            median_ms,
            min_ms: elapsed[0],
            tokens_per_second: tokens as f64 / (median_ms / 1_000.0),
        })
    }

    fn parse_args() -> Result<Args> {
        let mut parsed = Args::default();
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let mut value = || args.next().with_context(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--model" => parsed.model = PathBuf::from(value()?),
                "--tokens" => parsed.tokens = value()?.parse().context("invalid --tokens")?,
                "--warmup" => parsed.warmup = value()?.parse().context("invalid --warmup")?,
                "--iterations" => {
                    parsed.iterations = value()?.parse().context("invalid --iterations")?
                }
                // Cargo appends this marker for harness-less bench targets.
                "--bench" => {}
                "--help" | "-h" => {
                    println!(
                        "Usage: moe_layer [--model PATH] [--tokens N] [--warmup N] [--iterations N]"
                    );
                    std::process::exit(0);
                }
                _ => bail!("unknown argument {flag:?}"),
            }
        }
        if parsed.tokens == 0 || parsed.iterations == 0 {
            bail!("--tokens and --iterations must be positive");
        }
        Ok(parsed)
    }
}
