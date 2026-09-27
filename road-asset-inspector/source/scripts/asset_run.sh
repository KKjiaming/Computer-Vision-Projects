export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)
python tools/asset_inference.py
# python tools/asset_eval.py
