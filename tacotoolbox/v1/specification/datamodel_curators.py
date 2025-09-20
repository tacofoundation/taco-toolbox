from typing import List, Optional

import pydantic

from tacotoolbox.specification.datamodel_contacts import Contact


class Curators(pydantic.BaseModel):
    """This extension provides a way to define curators. Useful for
    measuring the quality of a dataset.

    Fields:
        curators (List[Contact]): A list of curators.
    """

    curators: Optional[List[Contact]] = None
