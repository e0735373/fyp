seed=99942
for i in {0..999}
do
	python generate.py -n 10 -s $seed > path-$(printf "%06d" $i).pddl
	seed=$((seed+1))
done
