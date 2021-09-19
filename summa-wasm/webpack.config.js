const path = require("path");
const CopyWebpackPlugin = require("copy-webpack-plugin");
const WasmPackPlugin = require("@wasm-tool/wasm-pack-plugin");

module.exports = {
    entry: {
        index: './pkg/index',
    },
    devtool: false,
    mode: "production",
    output: {
        path: path.resolve(__dirname, "dist"),
        filename: "[name].js",
        clean: true,
        library: {
          name: 'summa-wasm',
          type: 'umd',
        },
    },
    experiments: {
        asyncWebAssembly: true,
        topLevelAwait: true
    },
    plugins: [
        new WasmPackPlugin({
            crateDirectory: path.resolve(__dirname, "."),
            extraArgs: `--target web`,
            forceMode: 'production'
        }),
        new CopyWebpackPlugin({
            patterns: [{
                context: './src/',
                from: '**/*.ts',
                to: '../pkg'
            }]
        })
    ],
    module: {
        rules: [{
            test: /\.wasm(\.bin)?$/,
            type: "webassembly/async",
        }, {
            test: /\.tsx?$/,
            use: 'ts-loader',
            exclude: /node_modules/,
        }]
    },
    resolve: {
        extensions: [".js", ".ts"],
    },
};