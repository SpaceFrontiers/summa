//! Services responsible for various aspects a search engine like indices management or aliasing

pub(crate) mod beacon_service;
pub(crate) mod differential_updater;
pub(crate) mod index_service;

pub use beacon_service::BeaconService;
pub use differential_updater::DifferentialUpdater;
pub use index_service::IndexService;
