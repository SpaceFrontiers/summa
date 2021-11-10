import path from "path";
import {fileURLToPath} from 'url';
import WasmPackPlugin from '@wasm-tool/wasm-pack-plugin';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default {
    entry:  {
        'main': './src/index.ts',
        'worker': './src/web-index-service-worker.ts'
    },
    target: "web",
    output: {
        assetModuleFilename: '[name][ext]',
        clean: true,
        path: path.resolve(__dirname, 'dist'),
        filename: '[name].js',
        library: 'summa-wasm',
        libraryTarget: 'umd',
        umdNamedDefine: true
    },
    experiments: {
        topLevelAwait: true
    },
    module: {
        rules: [
            {
                test: /\.tsx?$/,
                use: 'ts-loader',
                exclude: /node_modules/,
            }
        ],
    },
    resolve: {
        extensions: ['.tsx', '.ts', '.js'],
    },
    plugins: [
        new WasmPackPlugin({
            crateDirectory: path.resolve(__dirname, 'crate'),

            // Check https://rustwasm.github.io/wasm-pack/book/commands/build.html for
            // the available set of arguments.
            //
            // Optional space delimited arguments to appear before the wasm-pack
            // command. Default arguments are `--verbose`.
            args: '--log-level warn',
            // Default arguments are `--typescript --target browser --mode normal`.
            extraArgs: "--target web --mode normal",

            // Optional array of absolute paths to directories, changes to which
            // will trigger the build.
            // watchDirectories: [
            //   path.resolve(__dirname, "another-crate/src")
            // ],

            // The same as the `--out-dir` option for `wasm-pack`
            // outDir: "pkg",

            // The same as the `--out-name` option for `wasm-pack`
            // outName: "index",

            // If defined, `forceWatch` will force activate/deactivate watch mode for
            // `.rs` files.
            //
            // The default (not set) aligns watch mode for `.rs` files to Webpack's
            // watch mode.
            // forceWatch: true,

            // If defined, `forceMode` will force the compilation mode for `wasm-pack`
            //
            // Possible values are `development` and `production`.
            //
            // the mode `development` makes `wasm-pack` build in `debug` mode.
            // the mode `production` makes `wasm-pack` build in `release` mode.
            // forceMode: "development",

            // Controls plugin output verbosity, either 'info' or 'error'.
            // Defaults to 'info'.
            // pluginLogLevel: 'info'
        })
    ]
}