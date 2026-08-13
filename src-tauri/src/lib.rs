use std::process::Command;
use std::sync::Mutex;
use tauri::{Manager, State}; // Added Manager trait here

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

pub struct SidecarProcess(pub Mutex<Option<std::process::Child>>);

#[derive(serde::Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram: u32,
    pub driver: String,
    pub cuda_version: String,
    pub detected: bool,
}

#[derive(serde::Serialize)]
pub struct SidecarStatus {
    pub running: bool,
    pub port: u16,
    pub pid: Option<u32>,
    pub message: String,
}

// REMOVED 'pub'
#[tauri::command]
fn detect_gpu() -> GpuInfo {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let raw = String::from_utf8_lossy(&out.stdout);
            let line = raw.lines().next().unwrap_or("").trim().to_string();
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

            if parts.len() >= 3 {
                let name = parts[0].to_string();
                let vram_mb: u32 = parts[1].parse().unwrap_or(0);
                let vram_gb = (vram_mb as f32 / 1024.0).round() as u32;
                let driver = parts[2].to_string();

                let cuda = Command::new("nvidia-smi")
                    .args(["--query-gpu=cuda_version", "--format=csv,noheader,nounits"])
                    .output()
                    .ok()
                    .and_then(|o| {
                        if o.status.success() {
                            Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| "Unknown".to_string());

                GpuInfo {
                    name,
                    vram: vram_gb,
                    driver,
                    cuda_version: cuda,
                    detected: true,
                }
            } else {
                no_gpu()
            }
        }
        _ => no_gpu(),
    }
}

// REMOVED 'pub'
#[tauri::command]
fn start_sidecar(
    state: State<SidecarProcess>,
    app_handle: tauri::AppHandle,
) -> SidecarStatus {
    let mut child_lock = state.0.lock().unwrap();

    // Check if the existing child is actually still alive
    if let Some(ref mut child) = *child_lock {
        match child.try_wait() {
            Ok(Some(_exit_status)) => {
                // Child has exited — clear the stale handle
                println!("[NeuralCut] Previous sidecar process has exited, clearing handle");
                *child_lock = None;
            }
            Ok(None) => {
                // Still running
                return SidecarStatus {
                    running: true,
                    port: 8188,
                    pid: Some(child.id()),
                    message: "Sidecar already running".to_string(),
                };
            }
            Err(_) => {
                // Error checking — assume dead, clear handle
                *child_lock = None;
            }
        }
    }

    // Kill any orphaned process holding port 8188 from a previous app run
    let _ = Command::new("cmd")
        .args(["/C", "for /f \"tokens=5\" %a in ('netstat -ano -p TCP ^| findstr \"127.0.0.1:8188\" ^| findstr \"LISTENING\"') do taskkill /F /PID %a"])
        .creation_flags(0x08000000) // CREATE_NO_WINDOW
        .output();
    // Brief pause to let the OS release the socket
    std::thread::sleep(std::time::Duration::from_millis(300));

    let resource_path = app_handle
        .path()
        .resource_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."));

    let current_dir = std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."));

    println!("[NeuralCut] current_dir: {:?}", current_dir);
    println!("[NeuralCut] resource_dir: {:?}", resource_path);

    // Try multiple possible paths
    let possible_pythons = vec![
        current_dir.join("src-tauri/sidecar/venv/Scripts/python.exe"),
        current_dir.join("sidecar/venv/Scripts/python.exe"),
        resource_path.join("sidecar/venv/Scripts/python.exe"),
    ];

    let possible_scripts = vec![
        current_dir.join("src-tauri/sidecar/main.py"),
        current_dir.join("sidecar/main.py"),
        resource_path.join("sidecar/main.py"),
    ];

    let python_path = possible_pythons.iter()
        .find(|p| {
            println!("[NeuralCut] Checking python path: {:?} exists={}", p, p.exists());
            p.exists()
        })
        .cloned();

    let script_path = possible_scripts.iter()
        .find(|p| {
            println!("[NeuralCut] Checking script path: {:?} exists={}", p, p.exists());
            p.exists()
        })
        .cloned();

    match (python_path, script_path) {
        (Some(python), Some(script)) => {
            println!("[NeuralCut] Spawning: {:?} {:?}", python, script);
            let mut cmd = Command::new(&python);
            cmd.arg(&script)
                .env("SIDECAR_PORT", "8188")
                .env("HF_TOKEN", std::env::var("HF_TOKEN").unwrap_or_else(|_| "".into()));

            if let Ok(models_dir) = std::env::var("MODELS_DIR") {
                cmd.env("MODELS_DIR", models_dir);
            }

            match cmd.spawn()
            {
                Ok(child) => {
                    let pid = child.id();
                    *child_lock = Some(child);
                    SidecarStatus {
                        running: true,
                        port: 8188,
                        pid: Some(pid),
                        message: "Sidecar started successfully".to_string(),
                    }
                }
                Err(e) => SidecarStatus {
                    running: false,
                    port: 8188,
                    pid: None,
                    message: format!("Failed to spawn sidecar: {}", e),
                },
            }
        }
        (python, script) => SidecarStatus {
            running: false,
            port: 8188,
            pid: None,
            message: format!(
                "Paths not found — python: {:?}, script: {:?}",
                python, script
            ),
        },
    }
}

// REMOVED 'pub'
#[tauri::command]
fn stop_sidecar(state: State<SidecarProcess>) -> bool {
    let mut child_lock = state.0.lock().unwrap();
    if let Some(mut child) = child_lock.take() {
        let _ = child.kill();
        true
    } else {
        false
    }
}

fn no_gpu() -> GpuInfo {
    GpuInfo {
        name: "No NVIDIA GPU detected".to_string(),
        vram: 0,
        driver: "N/A".to_string(),
        cuda_version: "N/A".to_string(),
        detected: false,
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(SidecarProcess(Mutex::new(None)))
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            detect_gpu,
            start_sidecar,
            stop_sidecar
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}