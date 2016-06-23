=========================
IEEE International Conference on Healthcare Informatics 2016 (ICHI 2016)
Healthcare Data Analytics Challenge
=========================


README.txt: Instructions about the data and submission file formats
=========================

This directory contains three text files, excluding this README file.
(a) ICHI2016-TrainData.tsv: contains the training data for the ICHI 2016 Healthcare Data Analytics Challenge
(b) ICHI2016-TestData.tsv: contains the test data for the ICHI 2016 Healthcare Data Analytics Challenge
(c) ICHI2016-run1.csv: shows a sample run of expected format in which the result runs should be submitted

File Formats:
(a) ICHI2016-TrainData.tsv
Each line consists of three tab-separated fields. The first field is the Category code (one of {DEMO, DISE, TRMT, GOAL, PREG, FMLY, SOCL}). 
The second and third fields are for the Title and the Question text, respectively. 
The first line is the header and is followed by 8,000 lines of training instances.

(b) ICHI2016-TestData.tsv
Each line consists of three tab-separated fields. The first field is the ID (runs from 1 to 3000). 
Similar to the training data, the second and third fields are for the Title and the Question text, respectively. 
The first line is the header and is followed by 3,000 lines of test instances that should be classified into one of the seven categories.

(c) ICHI2016-run1.csv
This file shows a sample classification run in the expected format.
Each line consists of two comma-separated fields. 
The first field is ID, and should match one of the IDs in the given test data (i.e. should be integers from 1 to 3000).
The second field is the Category code and should match one of the seven categories (i.e. one of {DEMO, DISE, TRMT, GOAL, PREG, FMLY, SOCL}). 

Include the first line header as "ID,Category". Typically, this will be followed by 3,000 lines of classification results.
Do not include two lines with the same ID. If the submitted files have lines with duplicate IDs, the first occurrence will be considered as final and the other occurrences will be ignored.
Do not include more than one category code. If multiple codes are included in a comma separated list, only the first code will be considered and others will be ignored.
Do not use a category code other than one of the seven specified (i.e. one of {DEMO, DISE, TRMT, GOAL, PREG, FMLY, SOCL}). If an unknown code is encountered, the instance will be treated as misclassification.

Name each submission run file uniquely as "<teamname>-run<ID>.csv", where <teamname> is the unique team name registered with the Challenge Chair, and <ID> is a number between 1 and 5. For example, if your registered teamname is "zenith" and it is the second run, the name of the submission file will be zenith-run2.csv

=========================
Contact Information:

If you have any questions or need additional information about the challenge, please contact the Healthcare Data Analytics Challenge Chair – Dr. V.G. Vinod Vydiswaran (vgvinodv@umich.edu)
=========================

