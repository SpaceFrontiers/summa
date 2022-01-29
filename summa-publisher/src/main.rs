use std::error::Error;
use std::str::FromStr;

use cid::Cid;
use clap::{arg, command};
use futures_lite::stream::StreamExt;
use iroh_rpc_types::store::StoreAddr;
use iroh_unixfs::unixfs::UnixfsNode;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
struct DataItem {
    pub cid: String,
    pub name: String,
}

#[derive(Deserialize, Serialize)]
struct StoreConfig {
    pub endpoint: String,
}

#[derive(Deserialize, Serialize)]
struct LayoutConfig {
    pub root: String,
    pub data: Vec<DataItem>,
}

#[derive(Deserialize, Serialize)]
struct Config {
    store: StoreConfig,
    layout: LayoutConfig,
}

static CHUNK_SIZE: usize = 1024 * 1024;

async fn publish(store_endpoint: &str, root_path: &str, data_paths: &[String]) -> Result<Option<Cid>, Box<dyn Error>> {
    let store_addr: StoreAddr = format!("irpc://{store_endpoint}").parse().expect("cannot parse");
    let store_client = iroh_rpc_client::store::StoreClient::new(store_addr).await.expect("cannot create store client");

    let root_directory = iroh_unixfs::builder::DirectoryBuilder::new()
        .chunker(iroh_unixfs::chunker::Chunker::Fixed(iroh_unixfs::chunker::Fixed { chunk_size: CHUNK_SIZE }))
        .add_path(root_path)
        .await?;
    let mut data_dir = iroh_unixfs::builder::DirectoryBuilder::new().name("data");

    for data_item in data_paths {
        let items: Vec<_> = data_item.split(':').collect();
        let (name, cid) = (items[0], items[1]);
        let cid = Cid::from_str(cid)?;
        let block = store_client.get(cid).await?.unwrap_or_else(|| panic!("Cannot find {cid} at store"));
        let node = UnixfsNode::decode(&cid, block)?;
        data_dir = data_dir.add_raw_block(iroh_unixfs::builder::RawBlock::new(name, node.encode(&cid::multihash::Code::Blake3_256)?));
    }
    let root_directory = root_directory.add_dir(data_dir.build()?)?.build()?;

    let mut blocks = root_directory.encode(&cid::multihash::Code::Blake3_256);
    let mut chunk = Vec::new();
    let mut chunk_size = 0u64;
    let mut cid = None;

    while let Some(block) = blocks.next().await {
        let block = block.expect("cannot get block");
        let block_size = block.data().len() as u64 + block.links().len() as u64 * 128;
        cid = Some(*block.cid());
        if chunk_size + block_size > CHUNK_SIZE as u64 {
            store_client.put_many(std::mem::take(&mut chunk)).await.expect("cannot_put");
            chunk_size = 0;
        }
        chunk.push(block.into_parts());
        chunk_size += block_size;
    }
    store_client.put_many(chunk).await.expect("cannot_put");
    Ok(cid)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let matches = command!()
        .name("summa-publisher")
        .override_usage("summa-publisher [OPTIONS] <SUBCOMMAND>")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .version(option_env!("CARGO_PKG_VERSION").unwrap_or("master"))
        .arg(arg!(-v --verbose ... "Level of verbosity"))
        .subcommand(
            command!("publish")
                .about("Publish data to storage")
                .arg(arg!(-s <STORE_ENDPOINT> "Iroh Store endpoint").default_value("0.0.0.0:4402").num_args(1))
                .arg(
                    arg!(-r <ROOT_PATH> "Path to files that will be put in the root of CAR")
                        .default_value("web/dist")
                        .num_args(1),
                )
                .arg(arg!(-d <DATA_PATH> "Names and cids of `data` directory members in format <name>:<cid>")),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("publish", submatches)) => {
            let store_endpoint = submatches.get_one::<String>("STORE_ENDPOINT").expect("wrong STORE_ENDPOINT");
            let root_path = submatches.get_one::<String>("ROOT_PATH").expect("wrong ROOT_PATH");
            let data_paths = submatches
                .get_many::<String>("DATA_PATH")
                .map(|x| x.cloned().collect::<Vec<String>>())
                .unwrap_or_default();
            let published_cid = publish(store_endpoint, root_path, &data_paths).await?.expect("no new cid").to_string();
            println!("{published_cid}");
        }
        _ => unreachable!("Exhausted list of subcommands and subcommand_required prevents `None`"),
    };
    Ok(())
}
