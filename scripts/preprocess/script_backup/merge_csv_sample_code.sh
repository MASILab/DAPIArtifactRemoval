for i in `ls | grep csv`;do cat $i >> final_combine_all.csv;done;
cat final_combine_all.csv | grep -v DICE_min > final_combine_all_v2.csv
# add header to the combine_all_v3.csv
