
seed=42
for i in {0..999}
do
	python generate.py -n 6 -p 3 -s $seed > sokoban-$(printf "%04d" $i).pddl
	seed=$((seed+1))
done
