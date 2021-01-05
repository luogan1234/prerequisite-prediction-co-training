for dataset in "chem"; do
  cmd="python build_graph.py -dataset $dataset"
  echo $cmd & $cmd
  cmd="python main.py -dataset $dataset -text_encoder bert -graph_layer gcn"
  echo $cmd & $cmd
done
