python ~/other/tensorflow/tensorflow/tensorflow/python/tools/freeze_graph.py \
    --input_graph=$1 \
    --input_checkpoint=$2 \
    --output_graph=$3 \
    --output_node_names=$4
