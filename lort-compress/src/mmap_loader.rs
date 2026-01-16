use anyhow::Result;
use candle_core::{Device, Tensor};
use memmap2::MmapOptions;
pub use safetensors::SafeTensors;
use std::fs::File;
use std::path::{Path, PathBuf};

pub struct MmapLoader {
    shards: Vec<MmapShard>,
}

struct MmapShard {
    _mmap: memmap2::Mmap, // Keep alive
    tensors: safetensors::SafeTensors<'static>,
    path: PathBuf,
}

impl MmapLoader {
    /// Initialize memory-mapped loader for one or more safetensors files
    pub fn new(paths: &[PathBuf]) -> Result<Self> {
        let mut shards = Vec::new();
        
        for path in paths {
            let file = File::open(path)?;
            let file_size = file.metadata()?.len();
            
            println!("  Memory-mapping: {} ({:.2} GB)", 
                     path.display(), 
                     file_size as f64 / 1_073_741_824.0);
            
            // SAFETY: We trust the file won't be modified during execution
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            
            // Parse safetensors header without loading data
            // SAFETY: Casting mmap bytes to static lifetime
            // This is safe because MmapShard holds the mmap alive
            let tensors = unsafe {
                let slice = std::slice::from_raw_parts(mmap.as_ptr(), mmap.len());
                SafeTensors::deserialize(slice)?
            };
            
            shards.push(MmapShard {
                _mmap: mmap,
                tensors: unsafe { std::mem::transmute(tensors) },
                path: path.clone(),
            });
        }
        
        Ok(Self { shards })
    }
    
    /// List all tensor names across all shards
    pub fn list_tensors(&self) -> Vec<String> {
        self.shards.iter()
            .flat_map(|s| s.tensors.names())
            .map(|n| n.to_string())
            .collect()
    }
    
    /// Load a specific tensor by name (pages into RAM on demand)
    pub fn load_tensor(&self, name: &str, device: &Device) -> Result<Tensor> {
        for shard in &self.shards {
            if let Ok(view) = shard.tensors.tensor(name) {
                // This triggers OS page-in for ONLY this tensor's bytes
                let shape = view.shape();
                let dtype = view.dtype().try_into()?;
                let data = view.data();
                
                let tensor = Tensor::from_raw_buffer(data, dtype, shape, device)
                    .map_err(|e| anyhow::anyhow!("Failed to create tensor: {}", e))?;
                return Ok(tensor);
            }
        }
        
        anyhow::bail!("Tensor '{}' not found in any shard", name)
    }
    
    /// Get total size across all shards
    pub fn total_size(&self) -> u64 {
        self.shards.iter()
            .map(|s| s._mmap.len() as u64)
            .sum()
    }
}

/// Estimate RAM needed to load model normally
pub fn estimate_ram_needed(paths: &[PathBuf]) -> Result<u64> {
    let total_size: u64 = paths.iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();
    
    // Add 20% overhead for tensor operations
    Ok((total_size as f64 * 1.2) as u64)
}

/// Get available system RAM
pub fn get_available_ram() -> Result<u64> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo")?;
        for line in meminfo.lines() {
            if line.starts_with("MemAvailable:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    // Value is in kB
                    return Ok(parts[1].parse::<u64>()? * 1024);
                }
            }
        }
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        // Conservative estimate: assume 8GB available
        return Ok(8 * 1024 * 1024 * 1024);
    }
    
    anyhow::bail!("Could not determine available RAM")
}
