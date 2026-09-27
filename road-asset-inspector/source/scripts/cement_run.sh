export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
# python tools/cement_inference.py
python tools/cement_eval8m.py
