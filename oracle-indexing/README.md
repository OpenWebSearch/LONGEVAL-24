```
docker build -t time-corpus-graph .
```

```
tira-run \
	--image time-corpus-graph \
	--input-dataset workshop-on-open-web-search/document-processing-20231027-training \
	--tira-vm-id ows \
	--push true
```

