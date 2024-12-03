mkdir data
cd data

gdown 1SjMOqzzKDtmkqmiesIyDy2zkEN7xGjbE
gdown 1BQvwhSoeDm-caCImtlbcAMzhI8MDsrCZ
gdown 1XZpEYxkPkDLU8U863ZBj1_7Qg7tLAybZ

unzip -q train_20241023.zip
unzip -q public\ test.zip
unzip -q Private_test_traffic.zip

mv public\ test public_test_images
mv private\ test private_test_images

rm train_20241023.zip
rm public\ test.zip
rm Private_test_traffic.zip

mv daytime/cam_10_000500.txt daytime/cam_10_00500.txt