# Memory

Agent memory interface implemented using LadybugDB. Other more optimized
interfaces are possible.

## Installation

```bash
uv sync
```

## Usage

```bash
uv pip install ladybug-memory
```

```python
from memory import LadybugMemory

mem = LadybugMemory("memory.lbdb")

mem.store("User prefers Python", memory_type="preference", importance=8)

results = mem.search("python")
for r in results:
    print(r.entry.content)
```
