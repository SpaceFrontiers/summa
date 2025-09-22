---
title: Development
parent: Extras
nav_order: 2
---

> Historical Summa 0.x guide. For Summa 2, use the [current documentation](https://github.com/SpaceFrontiers/summa/blob/main/docs/README.md).

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
