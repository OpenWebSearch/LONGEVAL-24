#docker build -t time-corpus-graph .
# A prepared image with python3.10, java 11, ir_datasets, tira, and PyTerrier installed 
FROM webis/ir-lab-wise-2023:0.0.4

# Update the tira command to use the latest version
RUN pip3 uninstall -y tira \
	&& pip3 install tira\>=0.0.125

ADD corpus-graph-over-time.py oracle-index.jsonl.gz /

ENTRYPOINT [ "/corpus-graph-over-time.py", "--input-dataset",  "$inputDataset", "--output-dir", "$outputDir" ]

