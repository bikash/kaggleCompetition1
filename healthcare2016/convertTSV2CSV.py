import csv
import sys
#def tsv2csv(file):
# python convertTSV2CSV.py < data/train.tsv > output.csv


tabin = csv.reader(sys.stdin, dialect=csv.excel_tab)
commaout = csv.writer(sys.stdout, dialect=csv.excel)
for row in tabin:
  commaout.writerow(row)