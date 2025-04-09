
start_id=500000
for i in {0..999}
do
	wall=$(((RANDOM % 31) + 10)) # 10-40
	python generate.py -id $((start_id+i)) > mysudoku-$(printf "%06d" $i).pddl
done
