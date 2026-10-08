use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{Manager, RunEvent, State};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// The running Python sidecar plus the port and auth token it was started with.
pub struct Sidecar {
    child: Option<Child>,
    port: u16,
    token: String,
}

pub struct SidecarState(pub Mutex<Sidecar>);

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
    pub token: String,
    pub pid: Option<u32>,
    pub message: String,
}

fn no_window(cmd: &mut Command) -> &mut Command {
    #[cfg(target_os = "windows")]
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd
}

#[tauri::command]
fn detect_gpu() -> GpuInfo {
    let output = no_window(Command::new("nvidia-smi").args([
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]))
    .output();

    if let Ok(out) = output {
        if out.status.success() {
            let raw = String::from_utf8_lossy(&out.stdout);
            let parts: Vec<String> = raw
                .lines()
                .next()
                .unwrap_or("")
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();
            if parts.len() >= 3 {
                let vram_mb: u32 = parts[1].parse().unwrap_or(0);
                let cuda = no_window(Command::new("nvidia-smi").args([
                    "--query-gpu=cuda_version",
                    "--format=csv,noheader,nounits",
                ]))
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .unwrap_or_else(|| "Unknown".to_string());
                return GpuInfo {
                    name: parts[0].clone(),
                    vram: (vram_mb as f32 / 1024.0).round() as u32,
                    driver: parts[2].clone(),
                    cuda_version: cuda,
                    detected: true,
                };
            }
        }
    }
    GpuInfo {
        name: "No NVIDIA GPU detected".to_string(),
        vram: 0,
        driver: "N/A".to_string(),
        cuda_version: "N/A".to_string(),
        detected: false,
    }
}

fn random_token() -> String {
    let mut buf = [0u8; 32];
    getrandom::getrandom(&mut buf).expect("OS random number generator unavailable");
    buf.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Ask the OS for a free loopback port instead of fighting over a fixed one.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or(47821)
}

fn status(s: &Sidecar, running: bool, message: &str) -> SidecarStatus {
    SidecarStatus {
        running,
        port: s.port,
        token: s.token.clone(),
        pid: s.child.as_ref().map(|c| c.id()),
        message: message.to_string(),
    }
}

#[tauri::command]
fn start_sidecar(state: State<SidecarState>, app_handle: tauri::AppHandle) -> SidecarStatus {
    let mut s = state.0.lock().unwrap();

    if let Some(child) = s.child.as_mut() {
        match child.try_wait() {
            Ok(None) => return status(&s, true, "Sidecar already running"),
            _ => s.child = None, // exited or unknown: start a fresh one
        }
    }

    let resource_dir = app_handle
        .path()
        .resource_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let roots = [cwd.join("src-tauri/sidecar"), cwd.join("sidecar"), resource_dir.join("sidecar")];
    let found = roots.iter().find(|r| {
        r.join("venv/Scripts/python.exe").exists() && r.join("main.py").exists()
    });
    let Some(root) = found else {
        return status(&s, false, "ThePipeline's AI engine files are missing. Please reinstall.");
    };

    let log_dir = app_handle
        .path()
        .app_local_data_dir()
        .unwrap_or_else(|_| std::env::temp_dir())
        .join("logs");
    let _ = std::fs::create_dir_all(&log_dir);
    // Append across restarts so a crash's log survives the automatic restart;
    // rotate once it passes 5 MB so it never grows without bound.
    let log_path = log_dir.join("sidecar.log");
    if std::fs::metadata(&log_path).map(|m| m.len() > 5 * 1024 * 1024).unwrap_or(false) {
        let _ = std::fs::rename(&log_path, log_dir.join("sidecar.old.log"));
    }
    let log = std::fs::OpenOptions::new().create(true).append(true).open(&log_path);

    s.port = free_port();
    s.token = random_token();

    let mut cmd = Command::new(root.join("venv/Scripts/python.exe"));
    cmd.arg(root.join("main.py"))
        .current_dir(root)
        .env("SIDECAR_PORT", s.port.to_string())
        .env("PIPELINE_TOKEN", &s.token)
        .env("PYTHONUNBUFFERED", "1")
        .env("PIPELINE_LOG_DIR", &log_dir)
        .stdin(Stdio::null());
    if let Ok(log) = log {
        if let Ok(err) = log.try_clone() {
            cmd.stdout(log).stderr(err);
        }
    }
    no_window(&mut cmd);

    match cmd.spawn() {
        Ok(child) => {
            s.child = Some(child);
            status(&s, true, "Sidecar started")
        }
        Err(e) => status(&s, false, &format!("Failed to start the AI engine: {}", e)),
    }
}

/// Kill the sidecar and everything it spawned (generation workers, ComfyUI).
fn kill_tree(s: &mut Sidecar) {
    if let Some(mut child) = s.child.take() {
        #[cfg(target_os = "windows")]
        {
            let _ = no_window(Command::new("taskkill").args(["/F", "/T", "/PID", &child.id().to_string()]))
                .output();
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[tauri::command]
fn stop_sidecar(state: State<SidecarState>) -> bool {
    let mut s = state.0.lock().unwrap();
    let was_running = s.child.is_some();
    kill_tree(&mut s);
    was_running
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .manage(SidecarState(Mutex::new(Sidecar {
            child: None,
            port: 0,
            token: String::new(),
        })))
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
        .invoke_handler(tauri::generate_handler![detect_gpu, start_sidecar, stop_sidecar])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            let state = handle.state::<SidecarState>();
            let mut s = state.0.lock().unwrap();
            kill_tree(&mut s);
        }
    });
}
