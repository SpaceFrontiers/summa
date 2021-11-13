---
title: Development
permalink: /development
---
Summa is armed with both Cargo and Bazel build systems. 
Feel free to use what is fit to you.

## Bazel Build

### Compile & Run
```bash
# Build main Summa binary with the search engine
bazel build summa-server
```

```bash
# Run Summa
bazel build summa-server
# or run with `release profile`
bazel build -c opt summa-server
```

## Integration Testing

```bash
# Launch all tests
bazel test //tests
```

## Publish

```bash
# Publish `aiosumma`
bazel build -c opt //aiosumma:aiosumma-wheel
twine upload bazel-bin/aiosumma/*.whl
```