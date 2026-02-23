use std::net::{IpAddr, SocketAddr};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::sync::Semaphore;
use tokio::time::timeout;
use ipnetwork::IpNetwork;
use std::sync::Arc;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    pub ip: IpAddr,
    pub port: u16,
    pub is_open: bool,
    pub latency_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanConfig {
    pub target: String, // IP or CIDR
    pub ports: Vec<u16>,
    pub timeout_ms: u64,
    pub concurrency: usize,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            target: "127.0.0.1".to_string(),
            ports: vec![80, 443, 22, 8080],
            timeout_ms: 1000,
            concurrency: 100,
        }
    }
}

pub struct NetworkScanner;

impl NetworkScanner {
    pub async fn scan(config: ScanConfig) -> Vec<ScanResult> {
        let targets = Self::parse_targets(&config.target);
        let semaphore = Arc::new(Semaphore::new(config.concurrency));
        let mut tasks = Vec::new();

        for ip in targets {
            for &port in &config.ports {
                let permit = semaphore.clone().acquire_owned().await.unwrap();
                let params = (ip, port, config.timeout_ms);
                
                tasks.push(tokio::spawn(async move {
                    let _permit = permit; // Drop permit when task completes
                    Self::scan_port(params.0, params.1, params.2).await
                }));
            }
        }

        let mut results = Vec::new();
        for task in tasks {
            if let Ok(res) = task.await {
                if res.is_open {
                    results.push(res);
                }
            }
        }
        results
    }

    fn parse_targets(target: &str) -> Vec<IpAddr> {
        if let Ok(network) = target.parse::<IpNetwork>() {
            network.iter().collect()
        } else if let Ok(ip) = target.parse::<IpAddr>() {
            vec![ip]
        } else {
            // Add DNS resolution logic here if needed, for now return empty or error
            eprintln!("Invalid target format: {}", target);
            vec![]
        }
    }

    async fn scan_port(ip: IpAddr, port: u16, timeout_ms: u64) -> ScanResult {
        let addr = SocketAddr::new(ip, port);
        let start = std::time::Instant::now();
        
        match timeout(Duration::from_millis(timeout_ms), TcpStream::connect(&addr)).await {
            Ok(Ok(_)) => ScanResult {
                ip,
                port,
                is_open: true,
                latency_ms: Some(start.elapsed().as_millis() as u64),
            },
            _ => ScanResult {
                ip,
                port,
                is_open: false,
                latency_ms: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scan_localhost() {
        // Start a listener on a random port
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        // Spawn listener task
        tokio::spawn(async move {
            let _ = listener.accept().await;
        });

        let config = ScanConfig {
            target: "127.0.0.1".to_string(),
            ports: vec![port, port + 1], // One open, one closed
            timeout_ms: 500,
            concurrency: 10,
        };

        let results = NetworkScanner::scan(config).await;
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].port, port);
        assert!(results[0].is_open);
    }
    
    #[test]
    fn test_cidr_parsing() {
        let ips = NetworkScanner::parse_targets("192.168.1.0/30");
        // 192.168.1.0, .1, .2, .3 = 4 addresses
        assert_eq!(ips.len(), 4);
    }
}
