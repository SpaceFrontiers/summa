use std::error::Error;
use std::str::FromStr;

use cid::Cid;
use clap::{arg, command};
use futures_lite::stream::StreamExt;
use iroh_rpc_types::store::StoreAddr;
use iroh_unixfs::unixfs::UnixfsNode;

static DEFAULT_CODE: cid::multihash::Code = cid::multihash::Code::Blake3_256;

async fn publish(store_endpoint: &str, root_path: &str, data_paths: &[String], chunk_size: usize) -> Result<Option<Cid>, Box<dyn Error>> {
    let store_addr: StoreAddr = format!("irpc://{store_endpoint}").parse()?;
    let store_client = iroh_rpc_client::store::StoreClient::new(store_addr).await?;

    let root_directory = iroh_unixfs::builder::DirectoryBuilder::new()
        .chunker(iroh_unixfs::chunker::Chunker::Fixed(iroh_unixfs::chunker::Fixed { chunk_size }))
        .add_path(root_path)
        .await?;
    let mut data_dir = iroh_unixfs::builder::DirectoryBuilder::new().name("data");

    for data_item in data_paths {
        let items: Vec<_> = data_item.split(':').collect();
        let (name, cid) = (items[0], items[1]);
        let cid = Cid::from_str(cid)?;
        let block = store_client.get(cid).await?.unwrap_or_else(|| panic!("`{cid}` is not found in Iroh Store"));
        let node = UnixfsNode::decode(&cid, block)?;
        data_dir = data_dir.add_raw_block(iroh_unixfs::builder::RawBlock::new(name, node.encode(&cid::multihash::Code::Blake3_256)?));
    }
    let root_directory = root_directory.add_dir(data_dir.build()?)?.build()?;

    let mut blocks = root_directory.encode(&DEFAULT_CODE);
    let mut chunk = Vec::new();
    let mut chunk_size = 0u64;
    let mut cid = None;

    while let Some(block) = blocks.next().await {
        let block = block?;
        let block_size = block.data().len() as u64 + block.links().len() as u64 * 128;
        cid = Some(*block.cid());
        if chunk_size + block_size > chunk_size {
            store_client.put_many(std::mem::take(&mut chunk)).await?;
            chunk_size = 0;
        }
        chunk.push(block.into_parts());
        chunk_size += block_size;
    }
    store_client.put_many(chunk).await?;
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
                .about("Tool composes CAR-file from a single directory and a list of CIDs that mounted under `/data` sub-directory")
                .arg(arg!(-s <STORE_ENDPOINT> "Iroh Store endpoint").default_value("0.0.0.0:4402").num_args(1))
                .arg(
                    arg!(-r <ROOT_PATH> "Path to files that will be put in the root of CAR")
                        .default_value("web/dist")
                        .num_args(1),
                )
                .arg(arg!(-d <DATA_PATH> "Names and CIDs for mounting under `data` sub-directory in format `<name>:<cid>`"))
                .arg(
                    arg!(-c <CHUNK_SIZE> "Size of a single block published to IPFS")
                        .default_value("1048576")
                        .num_args(1),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("publish", submatches)) => {
            let store_endpoint = submatches.get_one::<String>("STORE_ENDPOINT").expect("should be set");
            let root_path = submatches.get_one::<String>("ROOT_PATH").expect("should be set");
            let chunk_size = submatches.get_one::<usize>("CHUNK_SIZE").expect("should be set");
            let data_paths = submatches
                .get_many::<String>("DATA_PATH")
                .map(|x| x.cloned().collect::<Vec<String>>())
                .unwrap_or_default();
            let published_cid = publish(store_endpoint, root_path, &data_paths, *chunk_size)
                .await?
                .expect("no CID has been pubished, probably there is no data?")
                .to_string();
            println!("{published_cid}");
        }
        _ => unreachable!("Exhausted list of subcommands and subcommand_required prevents `None`"),
    };
    Ok(())
}
