
seed=42
for i in {1..1000}
do
	wall=$(((RANDOM % 31) + 10)) # 10-40
	./sokoban-generator-typed -n 8 -b 1 -w $wall -s $seed > sokoban-$i.pddl
	seed=$((seed+1))
done
