#dataset=$1
for dataset in "cs" "psy" "math"; do
  cmd="python build_graph.py -dataset $dataset"
  echo $cmd & $cmd
  for text_encoder in "lstm" "bert"; do
    for graph_layer in "gcn"; do
      cmd="python main.py -dataset $dataset -text_encoder $text_encoder -graph_layer $graph_layer"
      echo $cmd & $cmd
    done
  done
  for text_encoder in "bert"; do
    for graph_layer in "gcn"; do
      for init_num in 32 64 128 256 -1; do
        cmd="python main.py -dataset $dataset -text_encoder $text_encoder -graph_layer $graph_layer -init_num $init_num"
        echo $cmd & $cmd
      done
      for max_change_num in 0 36; do
        cmd="python main.py -dataset $dataset -text_encoder $text_encoder -graph_layer $graph_layer -max_change_num $max_change_num"
        echo $cmd & $cmd
      done
    done
  done
done
