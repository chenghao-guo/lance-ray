
# Distributed Index Building

Lance-Ray provides distributed index building functionality that leverages Ray's distributed computing capabilities to efficiently create text indices for Lance datasets. This is particularly useful for large-scale datasets as it can distribute index building work across multiple Ray worker nodes.

## New Distributed APIs

`create_scalar_index()` - Distributedly create scalar index index using ray. Currently only Inverted/FTS/BTREE are supported. Will add more index type support in the future.

### How It Works
The `create_scalar_index` function allows you to create full-text search indices for Lance datasets using the Ray distributed computing framework. This function distributes the index building process across multiple Ray worker nodes, with each node responsible for building indices for a subset of dataset fragments. These indices are then merged and committed as a single index.

**Backward Compatibility**:
   - Automatically detect availability of new APIs across different Lance versions
   - Gracefully fallback to raise tips when new APIs are unavailable


**`create_scalar_index`**

```python
def create_scalar_index(
    dataset: Union[str, "lance.LanceDataset"],
    column: str,
    index_type: Union[
        Literal["BTREE"],
        Literal["BITMAP"],
        Literal["LABEL_LIST"],
        Literal["INVERTED"],
        Literal["FTS"],
        Literal["NGRAM"],
        Literal["ZONEMAP"],
        IndexConfig,
    ],
    name: Optional[str] = None,
    *,
    replace: bool = True,
    train: bool = True,
    fragment_ids: Optional[list[int]] = None,
    fragment_uuid: Optional[str] = None,
    num_workers: int = 4,
    storage_options: Optional[dict[str, str]] = None,
    ray_remote_args: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> "lance.LanceDataset":

```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset` | `str` or `lance.LanceDataset` | Lance dataset or its URI |
| `column` | `str` | Column name to index |
| `index_type` | `str` or `IndexConfig` | Index type, can be `"INVERTED"`, `"FTS"`, `"BTREE"`, `"BITMAP"`, `"LABEL_LIST"`, `"NGRAM"`, `"ZONEMAP"`, or `IndexConfig` object |
| `name` | `str`, optional | Index name, auto-generated if not provided |
| `replace` | `bool`, optional | Whether to replace existing index with the same name, default is `True` |
| `train` | `bool`, optional | Whether to train the index, default is `True` |
| `fragment_ids` | `list[int]`, optional | Optional list of fragment IDs to build index on |
| `fragment_uuid` | `str`, optional | Optional fragment UUID for distributed indexing |
| `num_workers` | `int`, optional | Number of Ray worker nodes to use, default is 4 |
| `storage_options` | `Dict[str, str]`, optional | Storage options for the dataset |
| `ray_remote_args` | `Dict[str, Any]`, optional | Ray task options (e.g., `num_cpus`, `resources`) |
| `**kwargs` | `Any` | Additional arguments passed to `create_scalar_index` |

**Note:** For distributed indexing, currently only `"INVERTED"`,`"FTS"` and `"BTREE"` index types are supported.

### Return Value

The function returns an updated Lance dataset with the newly created index.


## Examples

### FTS Index
```python
import lance
import lance_ray as lr

# Create or load Lance dataset
dataset = lance.dataset("path/to/dataset")

# Build distributed index
updated_dataset = lr.create_scalar_index(
   dataset=dataset,
   column="text",
   index_type="INVERTED",
   num_workers=4
)

# Verify index creation
indices = updated_dataset.list_indices()
print(f"Index list: {indices}")

# Use index for search
results = updated_dataset.scanner(
   full_text_query="search term",
   columns=["id", "text"]
).to_table()
print(f"Search results: {results}")
```
### BTREE Index
```python
# Assume a LanceDataset with a numeric column "id" exists at this path
import lance_ray as lr

updated_dataset = lr.create_scalar_index(
    dataset="path/to/dataset",
    column="id",
    index_type="BTREE",
    name="btree_multiple_fragment_idx",
    replace=False,
    num_workers=4,
)

# Example queries
updated_dataset.scanner(filter="id = 100", columns=["id", "text"]).to_table()
updated_dataset.scanner(filter="id >= 200 AND id < 800", columns=["id", "text"]).to_table()
```


### Custom Index Name

```python
updated_dataset = lr.create_scalar_index(
   dataset="path/to/dataset",
   column="text",
   index_type="INVERTED",
   name="custom_text_index",
   num_workers=4
)
```

### Custom Ray Options

```python
updated_dataset = lr.create_scalar_index(
   dataset="path/to/dataset",
   column="text",
   index_type="INVERTED",
   num_workers=4,
   ray_remote_args={"num_cpus": 2, "resources": {"custom_resource": 1}}
)
```

### Index Replacement Control

```python
# Create index with custom name
updated_dataset = lr.create_scalar_index(
   dataset="path/to/dataset",
   column="text",
   index_type="INVERTED",
   name="my_text_index",
   num_workers=4
)

# Try to create another index with the same name (will replace by default)
updated_dataset = lr.create_scalar_index(
   dataset="path/to/dataset",
   column="text",
   index_type="INVERTED",
   name="my_text_index",  # Same name as before
   replace=True,          # Explicitly allow replacement (default behavior)
   num_workers=4
)

# Prevent index replacement
import lance_ray as lr

try:
    updated_dataset = lr.create_scalar_index(
       dataset="path/to/dataset",
       column="text",
       index_type="INVERTED",
       name="my_text_index",  # Same name as existing index
       replace=False,         # Prevent replacement
       num_workers=4
    )
except ValueError as e:
    print(f"Index creation failed: {e}")
    # Handle the error appropriately
```

### Performance Considerations

- For very large datasets, it's recommended to use more powerful CPU/memory ray worker nodes. Increasing `num_workers` can improve index building speed, but requires more computational nodes.
- Too many num_workers can cause large number of partitions, which cause FTS queries slowness as lots of index partitions need to be loaded when searching.
- If `num_workers` is greater than the number of fragments, it will be automatically adjusted to match the fragment count

### Important Notes

- **Index Type Support**: For distributed indexing, currently only `"INVERTED"`/`"FTS"`/`"BTREE"` index types are supported, even though the function signature accepts other index types.
- **Default Behavior**: The `replace` parameter defaults to `True`, meaning existing indices with the same name will be replaced without warning. Set `replace=False` to prevent accidental overwrites.
- **Fragment Selection**: Use `fragment_ids` parameter to build indices on specific fragments only. This is useful for incremental index building or testing.
- **Error Handling**: When `replace=False` and an index with the same name exists, a `ValueError` or `RuntimeError` will be raised depending on the execution context.
