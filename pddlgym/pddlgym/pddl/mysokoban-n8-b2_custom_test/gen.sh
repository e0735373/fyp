
seed=99942
for i in {1..100}
do
	wall=$(((RANDOM % 31) + 10)) # 10-40
	./a.out -n 8 -b 2 -w 0 -s 0 > sokoban-$i.pddl
	# wall=$(((RANDOM % 11) + 5)) # 5-15
	# ./sokoban-generator-typed -n 5 -b 1 -w $wall -s $seed > sokoban-$i.pddl
	seed=$((seed+1))
done
