mkdir data
cd data

gdown 1SjMOqzzKDtmkqmiesIyDy2zkEN7xGjbE
gdown 1BQvwhSoeDm-caCImtlbcAMzhI8MDsrCZ

unzip -q train_20241023.zip
unzip -q public\ test.zip

mv public\ test public_test_images

rm train_20241023.zip
rm public\ test.zip