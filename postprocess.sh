cmd="python build_graph.py -dataset moocen -no_course_dependency -no_user_act"
echo $cmd & $cmd
for seed in {0..49}; do
  cmd="python main.py -dataset moocen -model GCN -seed $seed -save_model"
  echo $cmd & $cmd
done
cmd="python postprocess.py -dataset moocen -model GCN"
echo $cmd & $cmd

cmd="python build_graph.py -dataset mooczh"
echo $cmd & $cmd
for seed in {0..49}; do
  cmd="python main.py -dataset mooczh -model GCN -seed $seed -save_model"
  echo $cmd & $cmd
done
cmd="python postprocess.py -dataset mooczh -model GCN"
echo $cmd & $cmd
