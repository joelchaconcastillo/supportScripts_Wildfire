counter=0
np=12
while IFS= read -r line
do
# echo $line
counter=$(($counter+1))
eval $line &
if ! (($counter % $np));
then
echo $counter
   wait
fi
done < "tasks"
