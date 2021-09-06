load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "16e9fca53ed6bd4ff4ad76facc9b7b651a89db1689a2877d6fd7b82aa824e366",
    urls = ["https://github.com/bazelbuild/rules_go/releases/download/v0.34.0/rules_go-v0.34.0.zip"],
)
http_archive(
    name = "bazel_gazelle",
    sha256 = "501deb3d5695ab658e82f6f6f549ba681ea3ca2a5fb7911154b5aa45596183fa",
    urls = ["https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.26.0/bazel-gazelle-v0.26.0.tar.gz"],
)
http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "320366665d19027cda87b2368c03939006a37e0388bfd1091c8d2a96fbc93bd8",
    strip_prefix = "grpc-1.48.1",
    urls = ["https://github.com/grpc/grpc/archive/v1.48.1.tar.gz"],
)
http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "b1e80761a8a8243d03ebca8845e9cc1ba6c82ce7c5179ce2b295cd36f7e394bf",
    urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.25.0/rules_docker-v0.25.0.tar.gz"],
)
http_archive(
    name = "rules_rust",
     sha256 = "6e507222f313fa675db241a2f8ceb6e1e64df2104d99b1ad4f10d8c3de0cb3f2",
    strip_prefix = "rules_rust-0.10.0",
    urls = ["https://github.com/bazelbuild/rules_rust/archive/0.10.0.tar.gz"],
)
http_archive(
    name = "com_google_protobuf",
    sha256 = "d7d204a59fd0d2d2387bd362c2155289d5060f32122c4d1d922041b61191d522",
    strip_prefix = "protobuf-3.21.5",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.21.5.tar.gz"],
)
http_archive(
    name = "rules_proto_grpc",
    sha256 = "bbe4db93499f5c9414926e46f9e35016999a4e9f6e3522482d3760dc61011070",
    strip_prefix = "rules_proto_grpc-4.2.0",
    urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/4.2.0.tar.gz"],
)
http_archive(
    name = "rules_python",
    strip_prefix = "rules_python-3a34e28c699cacf89e68d33187e8ca4e4fe01382",
    url = "https://github.com/ppodolsky/rules_python/archive/3a34e28c699cacf89e68d33187e8ca4e4fe01382.tar.gz",
)

# Proto
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

# GRPC
load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_repos", "rules_proto_grpc_toolchains")
rules_proto_grpc_toolchains()
rules_proto_grpc_repos()
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

# Rust
load("@rules_rust//rust:repositories.bzl", "rust_register_toolchains", "rules_rust_dependencies")
rules_rust_dependencies()
rust_register_toolchains(
    version="1.62.0",
)
load("@rules_rust//crate_universe:repositories.bzl", "crate_universe_dependencies")
crate_universe_dependencies(bootstrap = True)
load("@rules_rust//crate_universe:defs.bzl", "crate", "crates_repository", "render_config")
crates_repository(
    name = "crate_index",
    cargo_lockfile = "//:Cargo.lock",
    manifests = ["//:Cargo.toml"],
    generator = "@cargo_bazel_bootstrap//:cargo-bazel",
    annotations = {
        "rdkafka-sys": [crate.annotation(
            build_script_env = {"CARGO_MAKEFLAGS": ""},
        )],
        "prost-build": [crate.annotation(
            build_script_data = [
                "@com_google_protobuf//:protoc",
            ],
            build_script_env = {
                "PROTOC": "$(execpath @com_google_protobuf//:protoc)",
            },
        )],
        "tonic-reflection": [crate.annotation(
            build_script_data = [
                "@com_google_protobuf//:protoc",
            ],
            build_script_env = {
                "PROTOC": "$(execpath @com_google_protobuf//:protoc)",
            },
        )],
    }
)
load("@crate_index//:defs.bzl", "crate_repositories")
crate_repositories()

# Python
load("@rules_python//python:repositories.bzl", "python_register_toolchains")
python_register_toolchains(
    name = "python3_10",
    python_version = "3.10",
)

load("@python3_10//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "pip_parse")
pip_parse(
    name = "pip_modules",
    python_interpreter_target = interpreter,
    requirements_lock = "//aiosumma:requirements-lock.txt",
)
load("@pip_modules//:requirements.bzl", "install_deps")
install_deps()

# Proto / gRPC
load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_repos", "rules_proto_grpc_toolchains")
rules_proto_grpc_toolchains()
rules_proto_grpc_repos()

# Docker
load("@io_bazel_rules_docker//toolchains/docker:toolchain.bzl", docker_toolchain_configure="toolchain_configure")
docker_toolchain_configure(name="docker_config")
load("@io_bazel_rules_docker//repositories:repositories.bzl", container_repositories = "repositories")
container_repositories()
load("@io_bazel_rules_docker//repositories:deps.bzl", container_deps = "deps")
container_deps()
load("@io_bazel_rules_docker//container:container.bzl", "container_pull")
container_pull(
    name = "izihawa-base-image",
    digest = "sha256:878db474e52ad144ecf5faafb64c795989a83e3bbcccfaa005b743e51942a653",
    registry = "index.docker.io",
    repository = "izihawa/base-image",
)