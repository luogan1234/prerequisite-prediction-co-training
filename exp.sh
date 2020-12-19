for dataset in "moocen" "mooczh"; do
  cmd="python build_graph.py -dataset $dataset"
  echo $cmd & $cmd
  for text_encoder in "lstm" "bert" "bert-freeze"; do
    for graph_layer in "gcn" "gat"; do
      for init_num in 64 256 512; do
        cmd="python main.py -dataset $dataset -text_encoder $text_encoder -graph_layer $graph_layer -init_num $init_num"
        echo $cmd & $cmd
      done
      for max_change_num in 24 36 48; do
        cmd="python main.py -dataset $dataset -text_encoder $text_encoder -graph_layer $graph_layer -max_change_num $max_change_num"
        echo $cmd & $cmd
      done
    done
  done
done
