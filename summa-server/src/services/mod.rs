//! Services responsible for various aspects a search engine like indices management or aliasing

pub(crate) mod beacon_service;
pub(crate) mod index_service;

pub use beacon_service::BeaconService;
pub use index_service::IndexService;
