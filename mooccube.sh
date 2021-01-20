for dataset in "cs" "psy" "math"; do
  cmd="python build_graph.py -dataset $dataset"
  echo $cmd & $cmd
  for text_encoder in "lstm" "bert-freeze" "bert"; do
    for graph_layer in "gcn" "gat"; do
      for init_num in 32 64 128 256 -1; do
        cmd="python main.py -dataset $dataset -text_encoder $text_encoder -graph_layer $graph_layer -init_num $init_num -max_change_num 0"
        echo $cmd & $cmd
        cmd="python main.py -dataset $dataset -text_encoder $text_encoder -graph_layer $graph_layer -init_num $init_num"
        echo $cmd & $cmd
      done
    done
  done
done
