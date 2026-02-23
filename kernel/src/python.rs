use pyo3::prelude::*;
use crate::scanner::{NetworkScanner, ScanConfig, ScanResult};

#[pyclass]
#[derive(Clone)]
struct PyScanResult {
    #[pyo3(get)]
    ip: String,
    #[pyo3(get)]
    port: u16,
    #[pyo3(get)]
    is_open: bool,
    #[pyo3(get)]
    latency_ms: Option<u64>,
}

impl From<ScanResult> for PyScanResult {
    fn from(res: ScanResult) -> Self {
        PyScanResult {
            ip: res.ip.to_string(),
            port: res.port,
            is_open: res.is_open,
            latency_ms: res.latency_ms,
        }
    }
}

#[pyfunction]
fn scan_network(target: String, ports: Vec<u16>, timeout_ms: u64, concurrency: usize) -> PyResult<Vec<PyScanResult>> {
    let config = ScanConfig {
        target,
        ports,
        timeout_ms,
        concurrency,
    };

    // Create a new Tokio runtime for the scan
    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        NetworkScanner::scan(config).await
    });

    Ok(results.into_iter().map(PyScanResult::from).collect())
}

#[pymodule]
fn synos_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan_network, m)?)?;
    m.add_class::<PyScanResult>()?;
    Ok(())
}
