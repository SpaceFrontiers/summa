extern crate core;

use pyo3::prelude::*;
use pythonize::depythonize;
use summa_server::configs::server::ConfigHolder;
use summa_server::Server;

/*
   let core_config = summa_core::configs::core::ConfigBuilder::default()
       .doc_store_compress_threads(1)
       .writer_threads(None)
       .build()
       .expect("cannot build");
   let data_path = PathBuf::from(data_directory);
   let server_config = summa_server::configs::server::ConfigBuilder::default()
       .data_path(data_path.join("bin"))
       .logs_path(data_path.join("logs"))
       .core(core_config)
       .api(
           summa_server::configs::api::ConfigBuilder::default()
               .grpc_endpoint(endpoint.to_string())
               .build()
               .map_err(summa_core::Error::from)
               .unwrap(),
       )
       .build()
       .map_err(summa_core::Error::from)
       .unwrap();
*/

#[pyfunction]
fn summa_server_future<'a>(py: Python<'a>, config: &'a PyAny) -> PyResult<&'a PyAny> {
    let server_config: summa_server::configs::server::Config = depythonize(config).unwrap();
    let server_config_holder = ConfigHolder::from_config(server_config);
    let server = Server::from_server_config(server_config_holder);
    pyo3_asyncio::tokio::future_into_py(py, async move {
        server.run().await.unwrap();
        Ok(())
    })
}

/// A Python module implemented in Rust.
#[pymodule]
fn summa_embed_bin(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(summa_server_future, m)?)?;
    Ok(())
}
