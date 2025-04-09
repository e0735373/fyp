
seed=42
for i in {1..5000}
do
	wall=$(((RANDOM % 31) + 10)) # 10-40
	./sokoban-generator-typed -n 8 -b 2 -w $wall -s $seed > sokoban-$(printf "%06d" $i).pddl
	seed=$((seed+1))
done
