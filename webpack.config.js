const path = require('path');
const webpack = require('webpack');
const FileManagerPlugin = require('filemanager-webpack-plugin');


module.exports = env => {
    console.log(env);
    return {
        entry: './src/ts/index.ts',
        module: {
            rules: [
                {
                    test: /\.tsx?$/,
                    use: 'ts-loader',
                    exclude: /node_modules/,
                },
            ],
        },
        output: {
            publicPath: '/',
            filename: 'index.js',
            path: path.resolve(__dirname, 'static'),
        },
        resolve: {
            extensions: ['.js', '.json', '.vue', '.ts', 'js.map'],
            alias: {
                '@': path.resolve(__dirname),
                '~': path.resolve(__dirname),
            },
            modules: [
                'node_modules',
            ]
        },
        devtool: 'source-map',

        plugins: [
            new FileManagerPlugin({
                events: {
                    onStart: {},
                    onEnd: {
                        copy: [
                            { source: "static", destination: path.resolve(__dirname, 'dist/static') },
                            { source: "pkg", destination: path.resolve(__dirname, 'dist/pkg') },
                            { source: "app.py", destination: path.resolve(__dirname, 'dist/app.py') }
                        ],
                    },
                },
            }),
            //         new CopyPlugin({ patterns: [
            // {
            //     from: 'static',
            //     to: path.resolve(__dirname, 'dist/static'),
            //     toType: 'dir',
            // },
            // {
            //     from: 'pkg',
            //     to: path.resolve(__dirname, 'dist/pkg'),
            //     toType: 'dir',
            // },
            // {
            //     from: 'app.py',
            //     to: path.resolve(__dirname, 'dist'),
            //     toType: 'dir',
            // },

            // ]}),

            new webpack.DefinePlugin({
                APP: {
                    TITLE: JSON.stringify("l2s"),
                    AUTHOR: JSON.stringify("Error 404 <nice90sguy@gmail.com>"),
                }
            })
        ],
    };
}