sudo apt install zstd
wget -c https://the-eye.eu/public/AI/GPT-J-6B/step_383500_slim.tar.zstd
tar -I zstd -xf step_383500_slim.tar.zstd
sudo python3 -m pip uninstall tf-nightly tb-nightly -y
python3 -m pip install jax[tpu] jaxlib -f https://storage.googleapis.com/jax-releases/jax_releases.html
python3 -m pip install git+git@github.com:clashluke/mesh-transformer-jax
python3 -m pip install -r requirements.txt