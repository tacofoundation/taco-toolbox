{% set pit = collection.get("taco:pit_schema", collection.get("pit_schema", {})) %}
{% set field_schema = collection.get("taco:field_schema", collection.get("field_schema", {})) %}
{% set sources = collection.get("taco:sources", {}) %}
{% set extent = collection.get("extent", {}) %}

# {{ collection.get("title", id) }}

{% if description %}{{ description }}{% endif %}

## Dataset Information

**License**: {{ collection.get("licenses", ["Unknown"])|join(", ") }}

**Keywords**: {{ keywords|join(', ') if keywords else 'None' }}

**Tasks**: {{ collection.get("tasks", [])|join(', ') if collection.get("tasks") else 'None' }}

{% if sources %}
## Sources

**Partitions**: {{ sources["count"] }} file{% if sources["count"] != 1 %}s{% endif %}

{% if extent.get("spatial") %}
**Spatial coverage**: [{{ "%.2f"|format(extent["spatial"][0]) }}, {{ "%.2f"|format(extent["spatial"][1]) }}, {{ "%.2f"|format(extent["spatial"][2]) }}, {{ "%.2f"|format(extent["spatial"][3]) }}]
{% endif %}
{% if extent.get("temporal") %}
**Temporal coverage**: {{ extent["temporal"][0][:10] }} to {{ extent["temporal"][1][:10] }}
{% endif %}
{% endif %}

{% if pit %}
## Dataset Structure

**Root**: {{ pit.get("root", {}).get("type", "UNKNOWN") }} ({{ "{:,}".format(pit.get("root", {}).get("n", 0)) }} samples)

{% if pit.get("hierarchy") %}
**Hierarchy**:

{% for level, patterns in pit["hierarchy"].items()|sort %}
{% for pattern in patterns %}
- Level {{ level }}: {{ pattern.get("type", [])|join(" → ") }} ({{ "{:,}".format(pattern.get("n", 0)) }} samples)
{% endfor %}
{% endfor %}
{% endif %}
{% endif %}

{% if field_schema %}
## Metadata Fields

{% for level, fields in field_schema.items()|sort %}
### {{ level.upper() }}

| Field | Type | Description |
|-------|------|-------------|
{% for field in fields %}
| `{{ field[0] }}` | `{{ field[1] }}` | {% if field|length > 2 and field[2] %}{{ field[2]|replace("|", "\\|") }}{% endif %} |
{% endfor %}

{% endfor %}
{% endif %}

## Usage

### Python

```python
# pip install tacotoolbox
import tacoreader

ds = tacoreader.load("{{ id }}.tacozip")
print(f"Samples: {len(ds.data)}")
```

### R

```r
# install.packages("tacoreader")
library(tacoreader)

ds <- load_taco("{{ id }}.tacozip")
cat(sprintf("Samples: %d\n", nrow(ds$data)))
```

### Julia

```julia
# using Pkg; Pkg.add("TacoReader")
using TacoReader

ds = load_taco("{{ id }}.tacozip")
println("Samples: ", size(ds.data, 1))
```

{% if collection.get("curators") %}
## Curators

{% for curator in collection["curators"] %}
- **{{ curator.get("name", "Unknown") }}**{% if curator.get("organization") %} ({{ curator["organization"] }}){% endif %}
{% endfor %}
{% endif %}

---

Generated with [TacoToolbox](https://github.com/tacotoolbox/tacotoolbox)