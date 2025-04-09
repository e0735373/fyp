
seed=99942
initial_seed=$seed
for i in {0..999}
do
	while [ $(($seed - $initial_seed)) -eq 278 ] || [ $(($seed - $initial_seed)) -eq 998 ] || [ $(($seed - $initial_seed)) -eq 950 ] || [ $(($seed - $initial_seed)) -eq 422 ]; do
		seed=$((seed+1))
	done

	python generate.py -n 6 -p 3 -s $seed > sokoban-$(printf "%04d" $i).pddl
	seed=$((seed+1))
done
