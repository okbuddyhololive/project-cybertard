cd /home/ubuntu/codename-okbh/
export DISCORD_TOKEN=""
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
export XRT_TPU_CONFIG="localservice;0;localhost:51011" 
export XLA_FLAGS="--xla_force_host_platform_device_count=48" 
export TF_CPP_MIN_LOG_LEVEL=4 
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=60000000000

python3 bot.py