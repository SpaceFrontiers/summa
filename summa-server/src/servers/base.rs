use crate::errors::SummaServerResult;

pub trait BaseServer {
    fn set_listener(endpoint: &str) -> SummaServerResult<tokio::net::TcpListener> {
        let std_listener = std::net::TcpListener::bind(endpoint)?;
        std_listener.set_nonblocking(true)?;
        Ok(tokio::net::TcpListener::from_std(std_listener)?)
    }
}
