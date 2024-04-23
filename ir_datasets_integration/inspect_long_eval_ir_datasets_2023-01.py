import ir_datasets

data = ir_datasets.load('longeval/2023-01')
docs_store = data.docs_store()
print(docs_store.get('doc012301500007'))
# starts with:
#Clock\n- Howling Pixel
