::: easyimaging.sample_model.atoms.Atom

    options:
      heading_level: 2

::: easyimaging.sample_model.atoms.Atoms

    options:
      heading_level: 2

The full list of pre-defined
[`Atom`][easyimaging.sample_model.atoms.Atom] instances in the
[`Atoms`][easyimaging.sample_model.atoms.Atoms] enum is listed below.
<br> Isotopes with `nan` values indicate the natually occuring abundant
isotope.

```python exec="true"
from easyimaging.sample_model import Atoms
import pandas as pd
names = [f'**{atom.name}**' for atom in list(Atoms)]
pretty_table = pd.DataFrame(list(Atoms), index=names).drop('enum_id', axis=1)
print(pretty_table.rename_axis('name').to_markdown())
```
