export class WebIndexCoordinate {
  method: String;
  url_template: String;
  headers_template: {name: string, value: string}[] | null;
  files: Map<string, number>;
  enabled: boolean = true;
  constructor(
    method: String,
    url_template: String,
    headers_template: {name: string, value: string}[] | null,
    files: Map<string, number>,
  ) {
    this.method = method
    this.url_template = url_template
    this.headers_template = headers_template;
    this.files = files;
    this.enabled = true;
  }
}

export function create_web_index_ipfs_coordinate(ipfs_url: String, ipfs_path: String, files: Map<string, number>) {
  return new WebIndexCoordinate(
    "GET",
    `${ipfs_url}${ipfs_path}/{file_name}`,
    [{name: "range", value: "bytes={start}-{end}"}],
    files
  )
}
