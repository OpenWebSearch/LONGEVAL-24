# Custom ir_datasets Integration for LongEval 2024

**You only need ot look into this directory if you want to do some "non-standard" document/query parsing.

This directory contains the code for parsing queries and documents.
If you want to focus your work on new/different ways to parse the queries/documents (e.g., using the french text or a combination of english/french), this would be your starting point.

Test the integration with snippets like:

```
import ir_datasets

data = ir_datasets.load('longeval/train')
docs_store = data.docs_store()
print(docs_store.get('doc062201000001'))
```

