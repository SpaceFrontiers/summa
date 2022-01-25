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

async fn publish(config_path: &str) -> Result<Option<Cid>, Box<dyn Error>> {
    let config: Config = serde_yaml::from_slice(std::fs::read(config_path)?.as_slice())?;

    let store_addr: StoreAddr = format!("irpc://{}", config.store.endpoint).parse().expect("cannot parse");
    let store_client = iroh_rpc_client::store::StoreClient::new(store_addr).await.expect("cannot create store client");

    let root_directory = iroh_unixfs::builder::DirectoryBuilder::new()
        .chunker(iroh_unixfs::chunker::Chunker::Fixed(iroh_unixfs::chunker::Fixed { chunk_size: CHUNK_SIZE }))
        .add_path(&config.layout.root)
        .await?;
    let mut data_dir = iroh_unixfs::builder::DirectoryBuilder::new().name("data");

    for data_item in config.layout.data {
        let cid = Cid::from_str(&data_item.cid)?;
        let block = store_client.get(cid).await?.unwrap_or_else(|| panic!("Cannot find {cid} at store"));
        println!("Adding {cid:?} with size {}", block.len());
        let node = UnixfsNode::decode(&cid, block)?;
        data_dir = data_dir.add_raw_block(iroh_unixfs::builder::RawBlock::new(&data_item.name, node.encode()?));
    }
    let root_directory = root_directory.add_dir(data_dir.build().await?)?.build().await?;

    let mut blocks = root_directory.encode();
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
                .about("Publish layouted data to storage")
                .arg(arg!(-c <CONFIG_PATH> "Path to config").default_value("layout.yaml").num_args(1)),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("publish", submatches)) => {
            let config_path = submatches.try_get_one::<String>("CONFIG_PATH")?.expect("no config selected");
            let published_cid = publish(config_path).await?.expect("no new cid").to_string();
            println!("{published_cid}");
        }
        _ => unreachable!("Exhausted list of subcommands and subcommand_required prevents `None`"),
    };
    Ok(())
}
