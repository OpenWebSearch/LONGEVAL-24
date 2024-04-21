docker run --rm -ti -v /mnt/ceph/tira/state/ir_datasets/longeval/test-collection:/root/.ir_datasets/longeval/test-collection --entrypoint sh mam10eks/longeval2024-ir-dataset:0.0.1


/irds_cli.sh  --ir_datasets_id longeval/2023-06 --skip_qrels True --include_original False --output_dataset_path /tmp/2023-06

/irds_cli.sh  --ir_datasets_id longeval/2023-08 --skip_qrels True --include_original False --output_dataset_path /tmp/2023-08



ln -s /mnt/ceph/tira/data/publicly-shared-datasets/longeval-2023-08/documents.jsonl documents.jsonl



# Re-Ranking:
docker run --rm -ti \
	-v /mnt/ceph/tira/state/ir_datasets/longeval/test-collection:/root/.ir_datasets/longeval/test-collection \
	-v /mnt/ceph/tira/data/runs/longeval-2023-06-20240418-training/tira-ir-starter/2024-04-18-19-31-43/output:/mnt/ceph/tira/data/runs/longeval-2023-06-20240418-training/tira-ir-starter/2024-04-18-19-31-43/output:ro \
	-v /mnt/ceph/tira/data/runs/longeval-2023-08-20240418-training/tira-ir-starter/2024-04-18-19-35-50/output:/mnt/ceph/tira/data/runs/longeval-2023-08-20240418-training/tira-ir-starter/2024-04-18-19-35-50/output:ro \
	-v /mnt/ceph/tira/data/runs/longeval-2023-06-20240418-training/tira-ir-starter/2024-04-18-19-31-43-rerank-2024-04-19-11-41-33:/mnt/ceph/tira/data/runs/longeval-2023-06-20240418-training/tira-ir-starter/2024-04-18-19-31-43-rerank-2024-04-19-11-41-33 \
	-v /mnt/ceph/tira/data/runs/longeval-2023-08-20240418-training/tira-ir-starter/2024-04-18-19-35-50-rerank-2024-04-19-11-41-33:/mnt/ceph/tira/data/runs/longeval-2023-08-20240418-training/tira-ir-starter/2024-04-18-19-35-50-rerank-2024-04-19-11-41-33 \
	 --entrypoint sh mam10eks/longeval2024-ir-dataset:0.0.1




/irds_cli.sh --ir_datasets_id longeval/2023-06 --output_dataset_path /foo --include_original False --rerank /mnt/ceph/tira/data/runs/longeval-2023-06-20240418-training/tira-ir-starter/2024-04-18-19-31-43/output/run.txt


Empty: 337348/1790028

Empty: 1094132/2531614

