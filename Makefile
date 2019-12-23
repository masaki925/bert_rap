DOCKER_NAME=bert_rap
#APP_URL=http://localhost:5000/rap
APP_URL=http://34.84.14.181/rap
QUERY=おまえの母ちゃんでべそ

GCLOUD_PROJECT_ID=bert-rap
GCLOUD_APP_NAME=bert-rap

help:
	@echo "USAGE:"
	@echo "    make build"
	@echo "    make preprocess"
	@echo "    make train"
	@echo "    make generate"
	@echo "    ================"
	@echo "    make build_and_push_to_gcr"
	@echo "    make deploy_cloud_run"

build:
	docker build -t ${DOCKER_NAME} .

test:
	curl -v -H "Accept: application/json" -H "Content-type: application/json" -X POST -d "{ \"verse\": \"$(QUERY)\" }" $(APP_URL)


preprocess:
	docker run --rm -v ${PWD}:/app -w /app ${DOCKER_NAME}:latest python preprocess.py --input_file data/rap.txt --output_file data/out.txt --model /usr/local/libexec/jumanpp/jumandic.jppmdl

train:
	python ./bert_mouth.py \
		--bert_model ./Japanese_L-12_H-768_A-12_E-30_BPE_transformers/ \
		--output_dir ./models \
		--train_file tmp/train.txt \
		--valid_file tmp/valid.txt \
		--max_seq_length 128 \
		--do_train \
		--train_batch_size 10 \
		--num_train_epochs 100

generate:
	docker run --rm -v ${PWD}:/app -w /app jumanpp2:latest python ./bert_mouth.py \
		--bert_model ./rap_single_lines_model \
		--do_generate \
		--seq_length 20 \
		--max_iter 20

gcloud_set_project:
	gcloud config get-value project
	gcloud config set project ${GCLOUD_PROJECT_ID}
	gcloud config get-value project


build_and_push_to_gcr:
	gcloud builds submit --tag gcr.io/${GCLOUD_PROJECT_ID}/${GCLOUD_APP_NAME} --timeout=6000s

deploy_cloud_run:
	gcloud beta run deploy --image gcr.io/${GCLOUD_PROJECT_ID}/${GCLOUD_APP_NAME}

