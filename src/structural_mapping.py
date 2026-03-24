#!/usr/bin/env python3

from rcsbapi.data import DataQuery as Query

query = Query(
    input_type = "entries",
    input_ids = ["8JBR"],
    return_data_list = ["exptl.method", "struct.title"]
)

result = query.exec()

print(result)