//! CUDA microbenchmarks used to locate MoE routing and expert bottlenecks.

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
fn main() {
    eprintln!("moe_primitives requires Linux and --features training-fusion");
    std::process::exit(2);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn main() -> anyhow::Result<()> {
    benchmark::run()
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
mod benchmark {
    use std::time::Instant;

    use anyhow::{Context, Result, bail};
    use burn::prelude::*;
    use burn::tensor::activation::{silu, softmax};
    use burn::tensor::{DType, Distribution, IndexingUpdateOp, TensorData};
    use burn_nn::LinearConfig;
    use serde::Serialize;
    use summa_llm::default_device;

    #[derive(Debug)]
    struct Args {
        tokens: usize,
        hidden: usize,
        intermediate: usize,
        experts: usize,
        top_k: usize,
        warmup: usize,
        iterations: usize,
    }

    impl Default for Args {
        fn default() -> Self {
            Self {
                tokens: 8_192,
                hidden: 512,
                intermediate: 768,
                experts: 8,
                top_k: 2,
                warmup: 5,
                iterations: 20,
            }
        }
    }

    #[derive(Serialize)]
    struct Measurement {
        operation: &'static str,
        median_ms: f64,
        min_ms: f64,
    }

    #[derive(Serialize)]
    struct Report {
        device: String,
        dtype: &'static str,
        tokens: usize,
        hidden_size: usize,
        intermediate_size: usize,
        experts: usize,
        top_k: usize,
        warmup: usize,
        iterations: usize,
        measurements: Vec<Measurement>,
    }

    pub fn run() -> Result<()> {
        let args = parse_args()?;
        let device = default_device();
        device.seed(19);
        let input = Tensor::<2>::random([args.tokens, args.hidden], Distribution::Default, &device)
            .cast(DType::BF16);
        let router = LinearConfig::new(args.hidden, args.experts)
            .with_bias(false)
            .init(&device);
        let route_logits =
            Tensor::<2>::random([args.tokens, args.experts], Distribution::Default, &device);
        let routes = args.tokens * args.top_k;
        let route_order_data = (0..args.experts)
            .flat_map(|expert| {
                (expert..routes)
                    .step_by(args.experts)
                    .map(|route| route as i64)
            })
            .collect::<Vec<_>>();
        let routed_values =
            Tensor::<2>::random([routes, args.hidden], Distribution::Default, &device)
                .cast(DType::BF16);
        let expert_inputs = Tensor::<3>::random(
            [args.experts, routes / args.experts, args.hidden],
            Distribution::Default,
            &device,
        )
        .cast(DType::BF16);
        let up_weights = Tensor::<3>::random(
            [args.experts, args.hidden, args.intermediate * 2],
            Distribution::Default,
            &device,
        )
        .cast(DType::BF16);
        let down_weights = Tensor::<3>::random(
            [args.experts, args.intermediate, args.hidden],
            Distribution::Default,
            &device,
        )
        .cast(DType::BF16);
        let expert_linears = (0..args.experts)
            .map(|_| {
                (
                    LinearConfig::new(args.hidden, args.intermediate * 2)
                        .with_bias(false)
                        .init(&device),
                    LinearConfig::new(args.intermediate, args.hidden)
                        .with_bias(false)
                        .init(&device),
                )
            })
            .collect::<Vec<_>>();

        let mut measurements = Vec::new();
        measurements.push(measure(&args, &device, "router_matmul_topk", || {
            let logits = burn::tensor::module::linear(
                input.clone(),
                router.weight.val().cast(DType::BF16),
                None,
            )
            .cast(DType::F32);
            let (top_logits, indices) = logits.topk_with_indices(args.top_k, 1);
            std::hint::black_box((softmax(top_logits, 1), indices));
        })?);
        measurements.push(measure(&args, &device, "topk_and_host_indices", || {
            let (top_logits, indices) = route_logits.clone().topk_with_indices(args.top_k, 1);
            let weights = softmax(top_logits, 1);
            let host = indices
                .reshape([routes])
                .into_data()
                .convert::<i64>()
                .to_vec::<i64>()
                .expect("indices must be readable");
            std::hint::black_box((weights, host));
        })?);
        measurements.push(measure(&args, &device, "route_upload", || {
            let order = Tensor::<1, Int>::from_data(
                TensorData::new(route_order_data.clone(), [routes]),
                &device,
            );
            std::hint::black_box(order);
        })?);
        let route_order = Tensor::<1, Int>::from_data(
            TensorData::new(route_order_data.clone(), [routes]),
            &device,
        );
        let token_rows = route_order.div_scalar(args.top_k as i64);
        measurements.push(measure(&args, &device, "route_gather", || {
            std::hint::black_box(input.clone().select(0, token_rows.clone()));
        })?);
        measurements.push(measure(&args, &device, "route_scatter_add", || {
            std::hint::black_box(Tensor::zeros_like(&input).select_assign(
                0,
                token_rows.clone(),
                routed_values.clone(),
                IndexingUpdateOp::Add,
            ));
        })?);
        measurements.push(measure(
            &args,
            &device,
            "route_upload_gather_scatter",
            || {
                let order = Tensor::<1, Int>::from_data(
                    TensorData::new(route_order_data.clone(), [routes]),
                    &device,
                );
                let token_rows = order.clone().div_scalar(args.top_k as i64);
                let gathered = input.clone().select(0, token_rows.clone());
                let output = Tensor::zeros_like(&input).select_assign(
                    0,
                    token_rows,
                    routed_values.clone(),
                    IndexingUpdateOp::Add,
                );
                std::hint::black_box((gathered, output));
            },
        )?);
        measurements.push(measure(&args, &device, "batched_expert_ffn", || {
            let projected = expert_inputs.clone().matmul(up_weights.clone());
            let gate = silu(projected.clone().slice([
                0..args.experts,
                0..routes / args.experts,
                0..args.intermediate,
            ]));
            let values = projected.slice([
                0..args.experts,
                0..routes / args.experts,
                args.intermediate..args.intermediate * 2,
            ]);
            std::hint::black_box((gate * values).matmul(down_weights.clone()));
        })?);
        measurements.push(measure(&args, &device, "looped_expert_ffn", || {
            let mut outputs = Vec::with_capacity(args.experts);
            let per_expert = routes / args.experts;
            for (expert, (up, down)) in expert_linears.iter().enumerate() {
                let expert_input = expert_inputs
                    .clone()
                    .slice([expert..expert + 1, 0..per_expert, 0..args.hidden])
                    .squeeze_dim::<2>(0);
                let projected = burn::tensor::module::linear(
                    expert_input,
                    up.weight.val().cast(DType::BF16),
                    None,
                );
                let gate = silu(
                    projected
                        .clone()
                        .slice([0..per_expert, 0..args.intermediate]),
                );
                let values =
                    projected.slice([0..per_expert, args.intermediate..args.intermediate * 2]);
                outputs.push(burn::tensor::module::linear(
                    gate * values,
                    down.weight.val().cast(DType::BF16),
                    None,
                ));
            }
            std::hint::black_box(Tensor::cat(outputs, 0));
        })?);

        println!(
            "{}",
            serde_json::to_string_pretty(&Report {
                device: format!("{device:?}"),
                dtype: "bfloat16_tensor_cores_fp32_router",
                tokens: args.tokens,
                hidden_size: args.hidden,
                intermediate_size: args.intermediate,
                experts: args.experts,
                top_k: args.top_k,
                warmup: args.warmup,
                iterations: args.iterations,
                measurements,
            })?
        );
        Ok(())
    }

    fn measure(
        args: &Args,
        device: &Device,
        operation: &'static str,
        mut function: impl FnMut(),
    ) -> Result<Measurement> {
        for _ in 0..args.warmup {
            function();
            device.sync()?;
        }
        let mut elapsed = Vec::with_capacity(args.iterations);
        for _ in 0..args.iterations {
            device.sync()?;
            let started = Instant::now();
            function();
            device.sync()?;
            elapsed.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        elapsed.sort_by(f64::total_cmp);
        Ok(Measurement {
            operation,
            median_ms: if args.iterations % 2 == 0 {
                (elapsed[args.iterations / 2 - 1] + elapsed[args.iterations / 2]) / 2.0
            } else {
                elapsed[args.iterations / 2]
            },
            min_ms: elapsed[0],
        })
    }

    fn parse_args() -> Result<Args> {
        let mut parsed = Args::default();
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let mut value = || args.next().with_context(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--tokens" => parsed.tokens = value()?.parse().context("invalid --tokens")?,
                "--warmup" => parsed.warmup = value()?.parse().context("invalid --warmup")?,
                "--iterations" => {
                    parsed.iterations = value()?.parse().context("invalid --iterations")?
                }
                "--bench" => {}
                _ => bail!("unknown argument {flag:?}"),
            }
        }
        if parsed.tokens == 0
            || parsed.iterations == 0
            || !(parsed.tokens * parsed.top_k).is_multiple_of(parsed.experts)
        {
            bail!("tokens and iterations must be positive; routes must divide experts");
        }
        Ok(parsed)
    }
}
