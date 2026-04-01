# Distributed Compaction

As Lance datasets evolve over time (e.g., frequent appends / overwrites), they can accumulate many small fragments. Compaction rewrites fragments into fewer, larger fragments to improve scan and query performance.

Lance-Ray provides a distributed compaction workflow backed by Ray workers.

## `compact_files`

```python
compact_files(
    uri=None,
    *,
    table_id=None,
    compaction_options=None,
    num_workers=4,
    storage_options=None,
    namespace_impl=None,
    namespace_properties=None,
    ray_remote_args=None,
)
```

Compact files in a single Lance table (dataset) using distributed Ray workers.

**Parameters:**

- `uri`: Dataset URI to compact (either `uri` OR `namespace_impl` + `table_id` required)
- `table_id`: Table identifier as a list of strings (requires `namespace_impl`)
- `compaction_options`: Optional `lance.optimize.CompactionOptions` instance
- `num_workers`: Number of Ray workers to use (default: 4)
- `storage_options`: Optional storage configuration dictionary
- `namespace_impl`: Namespace implementation type (e.g., `"rest"`, `"dir"`)
- `namespace_properties`: Properties for connecting to the namespace
- `ray_remote_args`: Optional kwargs for Ray remote tasks

**Returns:** `CompactionMetrics` or `None` if no compaction tasks are needed.

## `compact_database`

```python
compact_database(
    *,
    database,
    namespace_impl,
    namespace_properties=None,
    compaction_options=None,
    num_workers=4,
    storage_options=None,
    ray_remote_args=None,
)
```

Compact all tables under a given database (namespace).

This function lists tables under `database` via the namespace API and runs `compact_files` on each table.

**Parameters:**

- `database`: Database (namespace) identifier as a list of path segments, e.g. `['my_database']`
- `namespace_impl`: Namespace implementation type (e.g., `"rest"`, `"dir"`)
- `namespace_properties`: Properties for connecting to the namespace
- `compaction_options`: Optional `lance.optimize.CompactionOptions` instance (applied to every table)
- `num_workers`: Number of Ray workers per table (default: 4)
- `storage_options`: Optional storage configuration dictionary
- `ray_remote_args`: Optional kwargs for Ray remote tasks

**Returns:** A list of dictionaries, one per table, with keys `table_id` (the full table identifier) and `metrics` (the compaction result, or `None` if no compaction was needed).

## Examples

### Compact a single table by URI

```python
import lance_ray as lr

metrics = lr.compact_files(
    uri="/path/to/table.lance",
    num_workers=4,
)
print(metrics)
```

### Compact a table via namespace

```python
import lance_ray as lr

metrics = lr.compact_files(
    uri=None,
    namespace_impl="dir",
    namespace_properties={"root": "/path/to/tables"},
    table_id=["my_table"],
    num_workers=2,
)
print(metrics)
```

### Compact an entire database

```python
from lance.optimize import CompactionOptions
import lance_ray as lr

results = lr.compact_database(
    database=["my_db"],
    namespace_impl="dir",
    namespace_properties={"root": "/path/to/tables"},
    compaction_options=CompactionOptions(target_rows_per_fragment=10000),
    num_workers=2,
)

for item in results:
    print(item["table_id"], item["metrics"])
```
