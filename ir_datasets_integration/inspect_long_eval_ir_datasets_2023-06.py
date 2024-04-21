import ir_datasets

data = ir_datasets.load('longeval/2023-06')
docs_store = data.docs_store()
print(docs_store.get('doc062303200291'))
# starts with:
#Steel Guitar
#Instrumentals
#| Pedal

