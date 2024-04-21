import ir_datasets

data = ir_datasets.load('longeval/2023-08')
docs_store = data.docs_store()
print(docs_store.get('doc082300800502'))
# starts with:
#Hotel Hotel Faro Centro

