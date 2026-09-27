export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
python tools/asphalt_inference.py
python tools/asphalt_eval8m.py
