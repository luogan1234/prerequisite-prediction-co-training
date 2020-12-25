for dataset in "moocen" "mooczh"; do
  cmd="python build_graph.py -dataset $dataset"
  echo $cmd & $cmd
  for text_encoder in "bert-freeze"; do
    for graph_layer in "gcn"; do
      for init_num in 16 32 64 128 256 512 -1; do
        cmd="python main.py -dataset $dataset -text_encoder $text_encoder -graph_layer $graph_layer -init_num $init_num"
        echo $cmd & $cmd
      done
      for max_change_num in 18 36 72 144 288; do
        cmd="python main.py -dataset $dataset -text_encoder $text_encoder -graph_layer $graph_layer -max_change_num $max_change_num"
        echo $cmd & $cmd
      done
    done
  done
done
