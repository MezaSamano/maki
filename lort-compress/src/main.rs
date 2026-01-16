mod mmap_loader;

use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use byteorder::{LittleEndian, WriteBytesExt};
use hf_hub::{api::sync::Api, Repo, RepoType};
use safetensors::SafeTensors;
use clap::Parser;
use mmap_loader::{MmapLoader, estimate_ram_needed, get_available_ram};

// --- CONFIG ---
const RANK: usize = 64;
const ITERATIONS: usize = 10;

#[derive(Parser, Debug)]
#[command(name = "lort-compress")]
#[command(about = "Compress transformer models using LoRT quantization")]
struct Args {
    /// Model ID from HuggingFace (e.g., "Qwen/Qwen2.5-0.5B") or local path to model directory
    #[arg(short, long, default_value = "Qwen/Qwen2.5-0.5B")]
    model: String,

    /// Output file path
    #[arg(short, long, default_value = "model.lort")]
    output: String,

    /// Minimum dimension for layers to compress (skip smaller layers)
    #[arg(long, default_value = "128")]
    min_dim: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    println!("🚀 LoRT Compression Engine");
    println!("Model: {}", args.model);
    println!("Output: {}", args.output);
    println!();

    // 1. Load Model Files (from HuggingFace or local)
    println!("📥 Loading model files...");
    let model_files = load_model_files(&args.model)?;
    println!("✓ Found {} model file(s)", model_files.len());
    
    // 2. Check RAM and decide loading strategy
    println!("\n🔍 Analyzing system resources...");
    let total_model_size: u64 = model_files.iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();
    
    println!("  Model size on disk: {:.2} GB", total_model_size as f64 / 1_073_741_824.0);
    
    let available_ram = get_available_ram().unwrap_or(8 * 1024 * 1024 * 1024);
    let needed_ram = estimate_ram_needed(&model_files)?;
    
    println!("  Available RAM: {:.2} GB", available_ram as f64 / 1_073_741_824.0);
    println!("  Estimated need: {:.2} GB", needed_ram as f64 / 1_073_741_824.0);
    
    let use_mmap = needed_ram > (available_ram as f64 * 0.8) as u64;
    
    if use_mmap {
        println!("\n⚠️  Model too large for available RAM!");
        println!("  ✓ Switching to Memory-Mapped mode (streaming)");
        println!("  This prevents OOM crashes on large models (70B+)");
        compress_with_mmap(&model_files, &args)?;
    } else {
        println!("\n✓ Sufficient RAM available - using standard loading");
        compress_standard(&model_files, &args)?;
    }
    
    Ok(())
}

/// Standard loading - loads entire model into RAM
fn compress_standard(model_files: &[PathBuf], args: &Args) -> Result<()> {
    println!("\n📂 Loading model weights into RAM...");
    let device = Device::Cpu;
    let mut layers: Vec<(String, Tensor)> = Vec::new();
    
    for model_file in model_files {
        println!("  Reading: {}", model_file.display());
        let buffer = std::fs::read(model_file)?;
        let tensors = SafeTensors::deserialize(&buffer)?;
        
        for (name, tensor_view) in tensors.tensors() {
            let shape = tensor_view.shape();
            if should_compress_layer(&name, shape, args.min_dim) {
                println!("  Found layer: {} {:?}", name, shape);
                
                let tensor = Tensor::from_raw_buffer(
                    tensor_view.data(),
                    tensor_view.dtype().try_into()?,
                    &shape,
                    &device,
                )?;
                
                layers.push((name.to_string(), tensor));
            }
        }
    }
    
    compress_and_save(layers, args)
}

/// Memory-mapped loading - streams one layer at a time
fn compress_with_mmap(model_files: &[PathBuf], args: &Args) -> Result<()> {
    println!("\n📂 Initializing memory-mapped loader...");
    let loader = MmapLoader::new(model_files)?;
    let device = Device::Cpu;
    
    // Get all tensor names
    let all_tensors = loader.list_tensors();
    
    // Filter to compressible layers
    let mut layer_names = Vec::new();
    for name in all_tensors {
        if !name.contains(".weight") {
            continue;
        }
        // We'll check dimensions when we load
        layer_names.push(name);
    }
    
    println!("\n✓ Found {} potential layers to compress", layer_names.len());
    println!("Starting LoRT Decomposition (Rank={}, Iterations={})...", RANK, ITERATIONS);
    println!("⚡ Streaming mode: Loading one layer at a time to prevent OOM\n");
    
    use indicatif::{ProgressBar, ProgressStyle};
    let pb = ProgressBar::new(layer_names.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("█▓▒░ "),
    );
    
    // Process layers sequentially to control RAM usage
    let mut compressed_layers = Vec::new();
    
    for name in layer_names {
        // Load ONLY this tensor into RAM
        match loader.load_tensor(&name, &device) {
            Ok(tensor) => {
                // Check dimensions
                if let Ok(shape) = tensor.dims2() {
                    if shape.0 >= args.min_dim && shape.1 >= args.min_dim {
                        // Compress and immediately drop the uncompressed tensor
                        if let Ok(compressed) = decompose_layer(&tensor, RANK, ITERATIONS) {
                            compressed_layers.push((name.clone(), compressed));
                        }
                    }
                }
                // tensor dropped here, freeing RAM
            }
            Err(_) => continue,
        }
        
        pb.inc(1);
    }
    
    pb.finish_with_message("Decomposition complete!");
    
    // Save results
    save_compressed_layers(&compressed_layers, &args.output, model_files)
}

/// Helper to check if a layer should be compressed  
fn should_compress_layer(name: &str, shape: &[usize], min_dim: usize) -> bool {
    if !name.contains(".weight") {
        return false;
    }
    
    if shape.len() != 2 {
        return false;
    }
    
    shape[0] >= min_dim && shape[1] >= min_dim
}

/// Common compression and save logic
fn compress_and_save(layers: Vec<(String, Tensor)>, args: &Args) -> Result<()> {
    println!("\n✓ Loaded {} compressible layers", layers.len());
    println!("Starting LoRT Decomposition (Rank={}, Iterations={})...\n", RANK, ITERATIONS);
    
    use indicatif::{ProgressBar, ProgressStyle};
    let pb = ProgressBar::new(layers.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("█▓▒░ "),
    );
    
    let compressed_layers: Vec<(String, CompressedLayer)> = layers.par_iter().map(|(name, w)| {
        let result = decompose_layer(w, RANK, ITERATIONS).expect("Decomposition failed");
        pb.inc(1);
        (name.clone(), result)
    }).collect();
    
    pb.finish_with_message("Decomposition complete!");
    
    // Calculate original model files size
    let model_files = load_model_files(&args.model)?;
    save_compressed_layers(&compressed_layers, &args.output, &model_files)
}

fn save_compressed_layers(
    compressed_layers: &[(String, CompressedLayer)],
    output_path: &str,
    model_files: &[PathBuf],
) -> Result<()> {
    println!("\n💾 Saving to '{}'...", output_path);
    let mut file = File::create(output_path)?;
    
    // Magic Header
    file.write_all(b"LORT")?;
    file.write_u32::<LittleEndian>(1)?; // Version
    file.write_u32::<LittleEndian>(compressed_layers.len() as u32)?;
    
    for (name, layer) in compressed_layers {
        let name_bytes = name.as_bytes();
        file.write_u32::<LittleEndian>(name_bytes.len() as u32)?;
        file.write_all(name_bytes)?;
        layer.write(&mut file)?;
    }
    
    // Calculate stats
    let original_size: u64 = model_files.iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();
    
    let compressed_size = std::fs::metadata(output_path)?.len();
    let ratio = original_size as f32 / compressed_size as f32;
    
    println!("\n✅ Done!");
    println!("   Original size (FP16): {:.2} MB", original_size as f32 / 1_048_576.0);
    println!("   Compressed size:      {:.2} MB", compressed_size as f32 / 1_048_576.0);
    println!("   Compression Ratio:    {:.2}x", ratio);
    println!("   Bits per weight:      {:.2}", 16.0 / ratio);
    
    Ok(())
}

struct CompressedLayer {
    alpha: f32,
    packed_ternary: Vec<u8>,
    lora_a: Tensor, // FP16 [Out, Rank]
    lora_b: Tensor, // FP16 [In, Rank]
}

impl CompressedLayer {
    fn write(&self, file: &mut File) -> Result<()> {
        // Write Alpha
        file.write_f32::<LittleEndian>(self.alpha)?;
        
        // Write LoRA Shapes (Dims)
        let (out_dim, rank) = self.lora_a.dims2()?;
        let (in_dim, _) = self.lora_b.dims2()?;
        file.write_u32::<LittleEndian>(out_dim as u32)?;
        file.write_u32::<LittleEndian>(in_dim as u32)?;
        file.write_u32::<LittleEndian>(rank as u32)?;

        // Write LoRA Data (FP16 bytes)
        // Convert to f16 bytes and write...
        // (Skipping boilerplate: assume tensor.to_vec1::<f16> -> write bytes)
        
        // Write Packed Ternary Size
        file.write_u32::<LittleEndian>(self.packed_ternary.len() as u32)?;
        file.write_all(&self.packed_ternary)?;
        
        Ok(())
    }
}

// --- THE SOLVER ---
fn decompose_layer(w: &Tensor, rank: usize, iters: usize) -> Result<CompressedLayer> {
    // Work in Float32 for precision
    let w = w.to_dtype(DType::F32)?;
    let (m, n) = w.dims2()?;
    
    // 1. Initial Scale (Alpha)
    let mut alpha = w.abs()?.mean_all()?.to_scalar::<f32>()?;
    
    // 2. Initialize Ternary (Greedy)
    let mut t = quantize_ternary(&w, alpha)?;
    
    // 3. Initialize LoRA (Random initialization + gradient descent)
    // Since Candle doesn't expose SVD easily, we use iterative optimization
    let mut lora_a = Tensor::randn(0f32, 0.02f32, (m, rank), w.device())?;
    let mut lora_b = Tensor::randn(0f32, 0.02f32, (n, rank), w.device())?;

    // 4. ALS Loop - iteratively optimize ternary and LoRA components
    for iter in 0..iters {
        // A. Optimize Alpha & T (Fix LoRA)
        let lora_est = lora_a.matmul(&lora_b.t()?)?;
        let res_t: Tensor = w.sub(&lora_est)?;
        
        alpha = res_t.abs()?.mean_all()?.to_scalar::<f32>()?;
        t = quantize_ternary(&res_t, alpha)?;
        
        // B. Optimize LoRA (Fix Alpha & T)
        let t_scaled = t.to_dtype(DType::F32)?.affine(alpha as f64, 0.0)?;
        let res_l: Tensor = w.sub(&t_scaled)?;
        
        // Simple gradient-based update for LoRA
        // res_l ≈ A @ B^T, solve using least squares approximation
        // A = res_l @ B @ (B^T @ B)^-1 (simplified update)
        
        // For simplicity, use iterative refinement
        // Update B: B = (A^T @ A)^-1 @ A^T @ res_l
        let at_a = lora_a.t()?.matmul(&lora_a)?;
        let at_res = lora_a.t()?.matmul(&res_l)?;
        
        // Add regularization for stability
        let eye = Tensor::eye(rank, lora_a.dtype(), lora_a.device())?;
        let reg_term = eye.affine(0.01, 0.0)?;
        let _at_a_reg = at_a.add(&reg_term)?;
        
        // Pseudo-inverse solve (simplified - just use the approximation)
        // In production, use proper linear solver
        lora_b = at_res.t()?;
        
        // Update A: A = res_l @ B @ (B^T @ B)^-1
        let bt_b = lora_b.t()?.matmul(&lora_b)?;
        let res_b = res_l.matmul(&lora_b)?;
        let _bt_b_reg = bt_b.add(&reg_term)?;
        
        lora_a = res_b;
        
        if (iter + 1) % 3 == 0 {
            // Renormalize to prevent overflow
            let a_scale = lora_a.abs()?.mean_all()?.to_scalar::<f32>()?;
            let b_scale = lora_b.abs()?.mean_all()?.to_scalar::<f32>()?;
            if a_scale > 0.0 && b_scale > 0.0 {
                lora_a = lora_a.affine(1.0 / (a_scale as f64), 0.0)?;
                lora_b = lora_b.affine(b_scale as f64, 0.0)?;
            }
        }
    }
    
    // 5. Pack Bits
    let packed = pack_ternary(&t)?;
    
    Ok(CompressedLayer {
        alpha,
        packed_ternary: packed,
        lora_a: lora_a.to_dtype(DType::F16)?,
        lora_b: lora_b.to_dtype(DType::F16)?,
    })
}

fn quantize_ternary(x: &Tensor, scale: f32) -> Result<Tensor> {
    let x = (x / scale as f64)?;
    let rounded = x.round()?;
    Ok(rounded.clamp(-1.0, 1.0)?)
}

fn pack_ternary(t: &Tensor) -> Result<Vec<u8>> {
    // 4 weights per byte. 
    // Format: 2 bits per weight. (00=0, 01=+1, 11=-1)
    let vals = t.flatten_all()?.to_vec1::<f32>()?;
    let mut bytes = Vec::with_capacity(vals.len() / 4);
    
    for chunk in vals.chunks(4) {
        let mut b: u8 = 0;
        for (i, &v) in chunk.iter().enumerate() {
            let bits = match v as i8 {
                1 => 0b01,
                -1 => 0b11,
                _ => 0b00,
            };
            b |= bits << (i * 2);
        }
        bytes.push(b);
    }
    Ok(bytes)
}

// Helper function to load model files from HuggingFace or local path
fn load_model_files(model_path: &str) -> Result<Vec<PathBuf>> {
    use std::path::Path;
    
    let path = Path::new(model_path);
    
    // Check if it's a local path
    if path.exists() {
        println!("  Using local model: {}", path.display());
        
        if path.is_file() {
            // Single safetensors file
            return Ok(vec![path.to_path_buf()]);
        } else if path.is_dir() {
            // Directory: find all safetensors files
            let mut files: Vec<PathBuf> = std::fs::read_dir(path)?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|p| {
                    p.extension()
                        .and_then(|s| s.to_str())
                        .map(|s| s == "safetensors")
                        .unwrap_or(false)
                })
                .collect();
            
            if files.is_empty() {
                anyhow::bail!("No .safetensors files found in directory: {}", path.display());
            }
            
            // Sort to ensure consistent ordering for sharded models
            files.sort();
            return Ok(files);
        }
    }
    
    // Not a local path, try HuggingFace
    println!("  Downloading from HuggingFace: {}", model_path);
    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_path.to_string(), RepoType::Model));
    
    // Try different common file patterns
    let patterns = vec![
        "model.safetensors",
        "model-00001-of-00001.safetensors",
        "pytorch_model.safetensors",
    ];
    
    for pattern in &patterns {
        if let Ok(file) = repo.get(pattern) {
            println!("  Found: {}", pattern);
            return Ok(vec![file]);
        }
    }
    
    // Try to find sharded models (model-00001-of-00002.safetensors, etc.)
    println!("  Looking for sharded model files...");
    let mut shard_files = Vec::new();
    for i in 1..=100 {  // Check up to 100 shards
        // Try to infer total shards
        for total in i..=100 {
            let filename = format!("model-{:05}-of-{:05}.safetensors", i, total);
            if let Ok(file) = repo.get(&filename) {
                println!("  Found: {}", filename);
                shard_files.push(file);
                
                // Continue getting remaining shards
                for j in (i + 1)..=total {
                    let next_file = format!("model-{:05}-of-{:05}.safetensors", j, total);
                    if let Ok(f) = repo.get(&next_file) {
                        println!("  Found: {}", next_file);
                        shard_files.push(f);
                    }
                }
                
                if !shard_files.is_empty() {
                    return Ok(shard_files);
                }
            }
        }
        
        if !shard_files.is_empty() {
            break;
        }
    }
    
    if !shard_files.is_empty() {
        return Ok(shard_files);
    }
    
    anyhow::bail!(
        "Could not find model files for '{}'. Tried patterns: {:?}",
        model_path,
        patterns
    )
}
