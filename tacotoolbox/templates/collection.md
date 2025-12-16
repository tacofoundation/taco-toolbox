{% set pit = collection.get("taco:pit_schema", collection.get("pit_schema", {})) %}
{% set field_schema = collection.get("taco:field_schema", collection.get("field_schema", {})) %}
{% set sources = collection.get("taco:sources", {}) %}
{% set extent = collection.get("extent", {}) %}

# {{ collection.get("title", id) }}

{% if description %}{{ description }}{% endif %}

## Dataset Information

**Version**: {{ collection.get("dataset_version", "1.0.0") }}

**License**: {{ collection.get("licenses", ["Unknown"])|join(", ") }}

**Keywords**: {{ keywords|join(', ') if keywords else 'None' }}

**Tasks**: {{ collection.get("tasks", [])|join(', ') if collection.get("tasks") else 'None' }}

{% if sources %}
## Dataset Overview

**Partitions**: {{ sources["count"] }} file{% if sources["count"] != 1 %}s{% endif %}

{% if extent.get("spatial") %}
**Spatial coverage**: [{{ "%.2f"|format(extent["spatial"][0]) }}, {{ "%.2f"|format(extent["spatial"][1]) }}, {{ "%.2f"|format(extent["spatial"][2]) }}, {{ "%.2f"|format(extent["spatial"][3]) }}] (WGS84)
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
# pip install tacoreader
import tacoreader

ds = tacoreader.load("{{ id }}.tacozip")
print(f"ID: {ds.id}")
print(f"Version: {ds.version}")
print(f"Samples: {len(ds.data)}")
```

### R

```r
# Coming soon: R support is planned but not yet available
# install.packages("tacoreader")
library(tacoreader)

ds <- load_taco("{{ id }}.tacozip")
cat(sprintf("ID: %s\n", ds$id))
cat(sprintf("Version: %s\n", ds$version))
cat(sprintf("Samples: %d\n", nrow(ds$data)))
```

### Julia

```julia
# Coming soon: Julia support is planned but not yet available
# using Pkg; Pkg.add("TacoReader")
using TacoReader

ds = load_taco("{{ id }}.tacozip")
println("ID: ", ds.id)
println("Version: ", ds.version)
println("Samples: ", size(ds.data, 1))
```

{% if collection.get("providers") %}
## Data Providers

{% for provider in collection["providers"] %}
- **{{ provider.get("name", "Unknown") }}**{% if provider.get("organization") %} ({{ provider["organization"] }}){% endif %}{% if provider.get("role") %} - *{{ provider["role"] }}*{% endif %}

{% endfor %}
{% endif %}

{% if collection.get("curators") %}
## Dataset Curators

{% for curator in collection["curators"] %}
- **{{ curator.get("name", "Unknown") }}**{% if curator.get("organization") %} ({{ curator["organization"] }}){% endif %}{% if curator.get("email") %} - {{ curator["email"] }}{% endif %}

{% endfor %}
{% endif %}

{% if collection.get("publications") %}
## Publications & Citations

If you use this dataset in your research, please cite:

{% for pub in collection["publications"] %}

**DOI**: {{ pub.get("doi", "No DOI") }}

{{ pub.get("citation", "No citation provided") }}
{% if pub.get("summary") %}

*{{ pub.get("summary") }}*
{% endif %}

---
{% endfor %}

### BibTeX

```bibtex
@dataset{ {{- collection.get("id", "dataset") -}} {{ collection.get("dataset_version", "2024").split(".")[0] }},
  title = { {{- collection.get("title", collection.get("id", "Dataset")) -}} },
  author = { 
    {%- if collection.get("curators") -%}
      {{ collection["curators"]|map(attribute='name')|join(' and ') }}
    {%- else -%}
      Unknown
    {%- endif -%}
  },
  year = { 
    {%- if extent.get("temporal") -%}
      {{ extent["temporal"][0][:4] }}
    {%- else -%}
      2024
    {%- endif -%}
  },
  version = { {{- collection.get("dataset_version", "1.0.0") -}} },
  publisher = { 
    {%- if collection.get("curators") and collection["curators"][0].get("organization") -%}
      {{ collection["curators"][0]["organization"] }}
    {%- else -%}
      Unknown
    {%- endif -%}
  }
}
```
{% endif %}

---

Generated with [TacoToolbox](https://github.com/tacotoolbox/tacotoolbox)